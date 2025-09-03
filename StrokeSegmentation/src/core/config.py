from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Stroke Segmentation"

    DEVICE: str = "cuda"
    SAVE_PATH: str = "models/unet_model_new.pth"

    PREDICT_MODEL: str = "models/unet_model_new.pth"
    PREDICT_INPUT: str = "src/test.jpg"

    def __str__(self) -> str:
        return (
            "\n----------------------------------------------------------"
            + f"\nDEVICE: {self.DEVICE}\nSAVE_PATH: {self.SAVE_PATH}\nPREDICT_MODEL: {self.PREDICT_MODEL}\nPREDICT_INPUT: {self.PREDICT_INPUT}"
            + "\n----------------------------------------------------------\n"
        )


settings = Settings()
