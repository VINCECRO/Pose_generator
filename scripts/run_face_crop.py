"""Recadrage des visages (tête + cou) depuis les images brutes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from tqdm import tqdm

from dataset_builder.face_cropper import FaceCropper

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détecte et recadre les visages (tête + cou) depuis data/raw/."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw"),
        help="Dossier source (défaut : data/raw)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/faces"),
        help="Dossier de sortie (défaut : data/faces)",
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("data/metrics"),
        help="Dossier des JSON de métriques pour le fallback landmarks (défaut : data/metrics)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Seuil de confiance détection visage (défaut : 0.3)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Sauvegarde des images annotées (bounding boxes) dans data/faces_debug/",
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
    debug_dir = Path("data/faces_debug") if args.debug else None

    metrics_dir = args.metrics_dir if args.metrics_dir.exists() else None
    if metrics_dir is None and not args.quiet:
        print(f"Info : dossier metrics '{args.metrics_dir}' absent, fallback landmarks désactivé.")

    image_paths = sorted(
        p for p in args.input.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    total = len(image_paths)
    saved = 0
    skipped = 0
    no_face = 0
    by_detector = 0
    by_landmarks = 0

    cropper = FaceCropper(min_confidence=args.confidence, metrics_dir=metrics_dir)

    for image_path in tqdm(image_paths, desc="Cropping faces", disable=args.quiet):
        crops = cropper.crop(image_path, debug_dir=debug_dir)

        if not crops:
            no_face += 1
            continue

        for crop in crops:
            suffix = f"_face{crop.face_index}" if len(crops) > 1 else "_face"
            out_path = args.output / (image_path.stem + suffix + ".jpg")

            if out_path.exists():
                skipped += 1
                continue

            cv2.imwrite(str(out_path), crop.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if crop.method == "landmarks":
                by_landmarks += 1
            else:
                by_detector += 1

    cropper.close()

    print(
        f"\nTerminé — {total} images traitées : "
        f"{saved} visages sauvegardés "
        f"(détecteur: {by_detector}, landmarks fallback: {by_landmarks}), "
        f"{no_face} sans visage, {skipped} déjà présents."
    )
    if args.debug:
        print(f"Images de debug : {debug_dir}/")


if __name__ == "__main__":
    main()
