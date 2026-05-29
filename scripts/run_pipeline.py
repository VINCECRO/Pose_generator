"""Point d'entrée CLI pour le pipeline principal."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_builder.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline d'extraction de squelettes OpenPose et de métriques de poses."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="Dossier source contenant les images (défaut : data/raw)",
    )
    parser.add_argument(
        "--skeletons",
        type=Path,
        default=Path("data/skeletons"),
        help="Dossier de sortie pour les squelettes (défaut : data/skeletons)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/metrics"),
        help="Dossier de sortie pour les métriques JSON (défaut : data/metrics)",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=Path("data/filtered"),
        help="Dossier pour les images retenues (défaut : data/filtered)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Seuil de confiance MediaPipe (défaut : 0.8)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux — pas de barre de progression",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Erreur : le dossier source '{args.input}' n'existe pas.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        input_dir=args.input,
        skeleton_dir=args.skeletons,
        metrics_dir=args.metrics,
        filtered_dir=args.filtered,
        confidence_threshold=args.confidence,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
