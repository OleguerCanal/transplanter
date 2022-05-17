import logging

from .block_module import BlockModule

def map_blocks(teacher_model : BlockModule,
               student_model : BlockModule,
               show : bool = True) -> None:
    len_teacher = len(teacher_model)
    len_student = len(student_model)
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    assignations = list(split(list(range(len_student)), len_teacher))
    mapping = {}
    for i, assigned in enumerate(assignations): 
        mapping[i] = assigned
    if show:
        logging.info("Mapping: " + str(mapping))
    return mapping