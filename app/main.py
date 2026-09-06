import os

from dotenv import load_dotenv

from services.pipeline_service import PipelineService

load_dotenv()


def main():

    service = PipelineService(

        api_key=os.getenv(
            "GEMINI_API_KEY"
        ),

        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash"
        )

    )

    result = service.run(
        "media/input/video.mp4"
    )

    print("=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    print(result["english"])


if __name__ == "__main__":
    main()