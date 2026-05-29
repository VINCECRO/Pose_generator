"""Détection des 478 landmarks faciaux sur les crops data/faces/ + overlay."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from tqdm import tqdm

from dataset_builder.face_landmark_detector import FaceLandmarkDetector
from dataset_builder.face_landmark_renderer import FaceLandmarkRenderer

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détecte les 478 landmarks faciaux sur data/faces/ et génère les overlays."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/faces"),
        help="Dossier source (défaut : data/faces)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/face_landmarks"),
        help="Dossier de sortie overlays + JSON (défaut : data/face_landmarks)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Seuil de confiance détection (défaut : 0.5)",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Ne pas sauvegarder les JSON de landmarks",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Mode silencieux",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Erreur : dossier source '{args.input}' introuvable.", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    total = len(image_paths)
    done = 0
    skipped = 0
    no_face = 0

    detector = FaceLandmarkDetector(min_confidence=args.confidence)
    renderer = FaceLandmarkRenderer()

    for image_path in tqdm(image_paths, desc="Face landmarks", disable=args.quiet):
        overlay_path = args.output / (image_path.stem + "_overlay.png")
        if overlay_path.exists():
            skipped += 1
            continue

        result = detector.detect(image_path)
        if result is None:
            no_face += 1
            continue

        original = cv2.imread(str(image_path))
        if original is None:
            no_face += 1
            continue

        overlay = renderer.render_on_photo(result, original)
        cv2.imwrite(str(overlay_path), overlay)

        # Canvas fond noir
        canvas = renderer.render(result, output_size=(512, 512))
        cv2.imwrite(str(args.output / (image_path.stem + "_mesh.png")), canvas)

        if not args.no_json:
            payload = {
                "source_image": image_path.name,
                "image_dimensions": {"width": result.image_width, "height": result.image_height},
                "landmarks": [
                    {
                        "x": round(lm.x, 6),
                        "y": round(lm.y, 6),
                        "z": round(lm.z, 6),
                        "x_px": int(lm.x * result.image_width),
                        "y_px": int(lm.y * result.image_height),
                    }
                    for lm in result.landmarks
                ],
                "blendshapes": result.blendshapes,
            }
            json_path = args.output / (image_path.stem + ".json")
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        done += 1

    detector.close()

    print(
        f"\nTerminé — {total} images : "
        f"{done} traitées, {no_face} sans visage, {skipped} déjà présentes."
    )


if __name__ == "__main__":
    main()
