
import logging

def log_model_blocks(teacher_blocks, student_blocks):
    logging.info("Teacher blocks:")
    for block in teacher_blocks:
        logging.info(block)

    logging.info("Student blocks:")
    for block in student_blocks:
        logging.info(block)
