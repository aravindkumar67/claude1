import argparse
from core.config import load_config
from core.pipeline import FallenObjectPipeline

def main():
    parser = argparse.ArgumentParser(description="Streamlit-free fallen-object pipeline")
    parser.add_argument("--source", default="0", help="Video path or camera index (default 0)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--display", action="store_true", help="Show OpenCV output window")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = FallenObjectPipeline(load_config(args.config), source)
    for result in pipeline.run():
        if args.display:
            import cv2
            cv2.imshow("Fallen Object POC", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                pipeline.stop()
                break
    if args.display:
        import cv2
        cv2.destroyAllWindows()
    print(f"Finished. Events: {len(pipeline.events)}")
    for event in pipeline.events:
        print(event)

if __name__ == "__main__":
    main()
