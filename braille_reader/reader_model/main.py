import torch
import numpy as np
from PIL import Image
from pathlib import Path
from os.path import join
from .utils.extract_dic import ExtractDic
from .post_processing.postprocess import *
from .post_processing.tranformations import *
from .utils.label_tools import int_to_unicode
from .braille_inference_impl import BraileInferenceImpl
from .models.orientation_attemps import OrientationAttempts
from .models.image_processor import BrailleImagePreprocessor
from .models.letter import AMHARIC_ALPHABET, VOWELS, SYMBOLS, NUMBERS

parent_path = str(Path(__file__).parent)


class BrailleRecognizer:
    def __init__(self, verbose=1):
        self.verbose = verbose

        inference_width = 1024
        params_fn = join(parent_path, "model", "weights", "param.txt")
        model_weights_fn = join(parent_path, "model", "weights", "model.t7")
        device = "cpu"

        if not torch.cuda.is_available() and device != "cpu":
            print("CUDA not availabel. CPU is used")
            device = "cpu"

        params = ExtractDic.load(params_fn, verbose=verbose)
        params.data.net_hw = (
            inference_width,
            inference_width,
        )

        params.data.batch_size = 1
        params.augmentation = ExtractDic(
            img_width_range=(inference_width, inference_width),
            stretch_limit=0.0,
            rotate_limit=0,
        )

        self.preprocessor = BrailleImagePreprocessor(params, mode="inference")

        self.impl = BraileInferenceImpl(
            params,
            model_weights_fn,
            device,
            [False for _ in range(64)],
            verbose=verbose,
        )
        self.impl.to(device)

    def recognize(
        self,
        img: str | Image.Image,
        find_orientation: bool = True,
        align_results: bool = True,
        gt_rects: list = [],
    ):
        """
        a function to read an image file and return the braille text associated with it
        :param img: can be 1) PIL.Image or 2) filename to image (.jpg etc.)
        :param find_orientation: boolean value indicating whether or not to find the best orientation of the img
        """
        if gt_rects:
            assert (
                find_orientation == False
            ), "Can't find orientation if gt_rects is not empty"

        if not isinstance(img, Image.Image):
            try:
                img = Image.open(img)
            except Exception as e:
                print("Can't open image " + str(img) + ": " + str(e))
                return None

        np_img = np.asarray(img)
        if len(np_img.shape) > 2 and np_img.shape[2] < 3:
            np_img = np_img[:, :, 0]
        aug_img, aug_gt_rects = self.preprocessor.preprocess_and_augment(
            np_img, gt_rects
        )
        aug_img = BrailleImagePreprocessor.unify_shape(aug_img)
        input_tensor = self.preprocessor.to_normalized_tensor(
            aug_img, device=self.impl.device
        )
        input_tensor_rotated = torch.tensor(0).to(self.impl.device)

        aug_img_rot = None
        if find_orientation:
            np_img_rot = np.rot90(np_img, 1, (0, 1))
            aug_img_rot = self.preprocessor.preprocess_and_augment(np_img_rot)[0]
            aug_img_rot = BrailleImagePreprocessor.unify_shape(aug_img_rot)
            input_tensor_rotated = self.preprocessor.to_normalized_tensor(
                aug_img_rot, device=self.impl.device
            )

        with torch.no_grad():
            (
                boxes,
                labels,
                scores,
                best_idx,
                err_score,
            ) = self.impl(
                input_tensor,
                input_tensor_rotated,
                find_orientation=find_orientation,
            )

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        lines = boxes_to_lines(
            boxes,
            labels,
        )

        self._refine_lines(lines)

        aug_img = Image.fromarray(
            aug_img if best_idx < OrientationAttempts.ROT90 else aug_img_rot
        )

        if best_idx in (OrientationAttempts.ROT180, OrientationAttempts.ROT270):
            aug_img = aug_img.transpose(Image.ROTATE_180)

        if align_results:
            hom = find_transformation(lines, (aug_img.width, aug_img.height))
            if hom is not None:
                aug_img = transform_image(aug_img, hom)
                boxes = transform_rects(boxes, hom)
                lines = boxes_to_lines(boxes, labels)
                self._refine_lines(lines)
                aug_gt_rects = transform_rects(aug_gt_rects, hom)

        else:
            hom = None

        out_braille = []
        for ln in lines:
            if ln.has_space_before:
                out_braille.append("")
            s = ""
            s_brl = ""
            for ch in ln.chars:
                s_brl += int_to_unicode(0) * ch.spaces_before + int_to_unicode(ch.label)
            out_braille.append(s_brl)

        translated_text = self._translate_to_text(out_braille)

        results_dict = {
            "image": aug_img,
            "best_idx": best_idx,
            "err_scores": list([ten.cpu().data.tolist() for ten in err_score]),
            "gt_rects": aug_gt_rects,
            "homography": hom.tolist() if hom is not None else hom,
            "scores": scores,
            "boxes": boxes,
            "lines": lines,
            "braille": out_braille,
            "text": translated_text,
        }

        return results_dict

    def _refine_lines(self, lines):
        """
        :param boxes:
        :return:
        """

        refine_coeffs = [0.083, 0.092, -0.083, -0.013]

        for ln in lines:
            for ch in ln.chars:
                h = ch.refined_box[3] - ch.refined_box[1]
                coefs = np.array(refine_coeffs)
                deltas = h * coefs
                ch.refined_box = (np.array(ch.refined_box) + deltas).tolist()

    def _translate_to_text(self, braille_texts: list[str]):
        translated_texts = []

        for text in braille_texts:
            translated_text = ""
            i = 0
            is_num = False
            while i < len(text):
                # Get the character and its corresponding vowel
                char = text[i]
                vowel = text[i + 1] if i + 1 < len(text) else " "

                if is_num:
                    if char == "\u2806" or not char in NUMBERS:
                        is_num = False
                        if char == "\u2806":
                            i += 1
                            continue
                    else:
                        translated_text += NUMBERS[char]
                        i += 1
                        continue

                # Translate the character and vowel to Amharic
                elif char == "\u2800":
                    translated_text += " "
                    i += 1
                    continue

                elif char == "\u283c":  # number
                    is_num = True
                    i += 1
                    continue

                elif char in SYMBOLS:
                    translated_text += (
                        SYMBOLS[char]
                        if isinstance(SYMBOLS[char], str)
                        else SYMBOLS[char].get(vowel, "?")
                    )
                    if isinstance(SYMBOLS[char], str):
                        i += 1
                    else:
                        i += 2
                    continue

                elif char in AMHARIC_ALPHABET:
                    if vowel in VOWELS:
                        translated_text += AMHARIC_ALPHABET[char][vowel]
                        i += 2  # Move to the next character and vowel
                        continue
                    else:
                        translated_text += AMHARIC_ALPHABET[char][" "]
                        i += 1
                        continue
                else:
                    translated_text += (
                        char  # If no translation is found, keep the original character
                    )
                    i += 1
                    continue

            translated_texts.append(translated_text)

        return translated_texts


if __name__ == "__main__":
    img_filename_mask = join(parent_path, "photo_2023-12-16_11-14-23.jpg")

    verbose = 0

    recognizer = BrailleRecognizer(verbose=verbose)
    res: dict = recognizer.recognize(
        img_filename_mask,
        find_orientation=True,
        align_results=True,
    )

    with open("braille.txt", "w") as f:
        f.write("\n".join(res["text"]))
