# Transplanter

**Story**: I have a "small" ANN (~100M params) working decently ok, but I wanna see if a larger ANN (~400M params) can significantly outperform it. However, training from sratch models of these sizes is extremely expensive, so I need a way of efficiently transferring knowledge from one to the other.
Introducing **Transplanter**: a way of initializing the  weights such that new models can hit the ground running 🏃

**What is this repo solving?**

You have a trained model `teacher_model` but want to experiment architectural changes in a new model `student_model`. However, training from scratch is extremely expensive, so you need a way of initializing the new model's weights.

**How is that different from model distillation?**

In distillation you want to reduce a model size, in this case I'll be focussing on increasing it.

## Interface

```python
from transplanter import Transplanter

dataset = # read dataset
teacher_model = # read trained model
student_model = # instantiate new model

transplanter = Transplanter()
transplanter.transplant(
            from=teacher_model,
            to=student_model,
            dataset=dataset)

# here student_model should be performing almost as good as teacher_model
```

**NOTE:** This is a very early-stage work-in-process work. I will prioritize it for what I need now, but try to make it general enough to be applied to other PyTorch projects.
