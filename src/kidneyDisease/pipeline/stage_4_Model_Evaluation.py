
from kidneyDisease.config.configuration import ConfigurationManager
from kidneyDisease.components.Model_Evaluation import Evaluation,EvaluationConfig
from kidneyDisease import logger

STAGE_NAME = "Model Evaluation With mlflow "




class EvaluationPipline:
  def __init__(self) -> None:
        pass
  def main(self):
    config = ConfigurationManager()
    eval_config = config.get_evaluation_config()
    evaluation = Evaluation(eval_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()

    evaluation.save_score()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e