# DeepXDE Complete API Documentation

This document contains the complete API reference for DeepXDE library, automatically scraped from the official documentation.

**Total modules documented:** 15

**Total functions:** 0
**Total classes:** 107

---

## Table of Contents

- [deepxde](#deepxde) (0 functions, 14 classes)
- [deepxde.backend](#deepxde-backend) (0 functions, 0 classes)
- [deepxde.data](#deepxde-data) (0 functions, 27 classes)
- [deepxde.geometry](#deepxde-geometry) (0 functions, 18 classes)
- [deepxde.gradients](#deepxde-gradients) (0 functions, 0 classes)
- [deepxde.icbc](#deepxde-icbc) (0 functions, 10 classes)
- [deepxde.nn](#deepxde-nn) (0 functions, 1 classes)
- [deepxde.nn.jax](#deepxde-nn-jax) (0 functions, 3 classes)
- [deepxde.nn.paddle](#deepxde-nn-paddle) (0 functions, 7 classes)
- [deepxde.nn.pytorch](#deepxde-nn-pytorch) (0 functions, 8 classes)
- [deepxde.nn.tensorflow](#deepxde-nn-tensorflow) (0 functions, 6 classes)
- [deepxde.nn.tensorflow_compat_v1](#deepxde-nn-tensorflow_compat_v1) (0 functions, 11 classes)
- [deepxde.optimizers](#deepxde-optimizers) (0 functions, 0 classes)
- [deepxde.optimizers.pytorch](#deepxde-optimizers-pytorch) (0 functions, 1 classes)
- [deepxde.utils](#deepxde-utils) (0 functions, 1 classes)

---

## deepxde

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.html)

### Classes

#### deepxde.callbacks.Callback

```python
class deepxde.callbacks.Callback
```

Bases: object

**Methods:**

##### `init`

```python
init()
```

Init after setting a model.


##### `on_batch_begin`

```python
on_batch_begin()
```

Called at the beginning of every batch.


##### `on_batch_end`

```python
on_batch_end()
```

Called at the end of every batch.


##### `on_epoch_begin`

```python
on_epoch_begin()
```

Called at the beginning of every epoch.


##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_predict_begin`

```python
on_predict_begin()
```

Called at the beginning of prediction.


##### `on_predict_end`

```python
on_predict_end()
```

Called at the end of prediction.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


##### `set_model`

```python
set_model(model)
```

**Parameters:**

- `model`


---

#### deepxde.callbacks.CallbackList

```python
class deepxde.callbacks.CallbackList(callbacks=None)
```

Bases: Callback

**Parameters:**

- `callbacks` (optional, default: `None`)

**Methods:**

##### `append`

```python
append(callback)
```

**Parameters:**

- `callback`


##### `on_batch_begin`

```python
on_batch_begin()
```

Called at the beginning of every batch.


##### `on_batch_end`

```python
on_batch_end()
```

Called at the end of every batch.


##### `on_epoch_begin`

```python
on_epoch_begin()
```

Called at the beginning of every epoch.


##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_predict_begin`

```python
on_predict_begin()
```

Called at the beginning of prediction.


##### `on_predict_end`

```python
on_predict_end()
```

Called at the end of prediction.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


##### `set_model`

```python
set_model(model)
```

**Parameters:**

- `model`


---

#### deepxde.callbacks.DropoutUncertainty

```python
class deepxde.callbacks.DropoutUncertainty(period=1000)
```

Bases: Callback

**Parameters:**

- `period` (optional, default: `1000`)

**Methods:**

##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


---

#### deepxde.callbacks.EarlyStopping

```python
class deepxde.callbacks.EarlyStopping(min_delta=0, patience=0, baseline=None, monitor='loss_train', start_from_epoch=0)
```

Bases: Callback

**Parameters:**

- `min_delta` (optional, default: `0`): Minimum change in the monitored quantity
to qualify as an improvement, i.e. an absolute
change of less than min_delta, will count as no
improvement.
- `patience` (optional, default: `0`): Number of epochs with no improvement
after which training will be stopped.
- `baseline` (optional, default: `None`): Baseline value for the monitored quantity to reach.
Training will stop if the model doesn’t show improvement
over the baseline.
- `monitor` (optional, default: `'loss_train'`): The loss function that is monitored. Either ‘loss_train’ or ‘loss_test’
- `start_from_epoch` (optional, default: `0`): Number of epochs to wait before starting
to monitor improvement. This allows for a warm-up period in which
no improvement is expected and thus training will not be stopped.

**Methods:**

##### `get_monitor_value`

```python
get_monitor_value()
```


##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


---

#### deepxde.callbacks.FirstDerivative

```python
class deepxde.callbacks.FirstDerivative(x, component_x=0, component_y=0, period=1, filename=None, precision=2)
```

Bases: OperatorPredictor

**Parameters:**

- `x`: The input data.
- `component_x` (optional, default: `0`)
- `component_y` (optional, default: `0`)
- `period` (optional, default: `1`)
- `filename` (optional, default: `None`)
- `precision` (optional, default: `2`)

---

#### deepxde.callbacks.ModelCheckpoint

```python
class deepxde.callbacks.ModelCheckpoint(filepath, verbose=0, save_better_only=False, period=1, monitor='train loss')
```

Bases: Callback

**Parameters:**

- `filepath`
- `verbose` (optional, default: `0`): Verbosity mode, 0 or 1.
- `save_better_only` (optional, default: `False`): If True, only save a better model according to the quantity
monitored. Model is only checked at validation step according to
display_every in Model.train.
- `period` (optional, default: `1`): Interval (number of epochs) between checkpoints.
- `monitor` (optional, default: `'train loss'`): The loss function that is monitored. Either ‘train loss’ or ‘test loss’.

**Methods:**

##### `get_monitor_value`

```python
get_monitor_value()
```


##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


---

#### deepxde.callbacks.MovieDumper

```python
class deepxde.callbacks.MovieDumper(filename, x1, x2, num_points=100, period=1, component=0, save_spectrum=False, y_reference=None)
```

Bases: Callback

**Parameters:**

- `filename`
- `x1`
- `x2`
- `num_points` (optional, default: `100`)
- `period` (optional, default: `1`)
- `component` (optional, default: `0`)
- `save_spectrum` (optional, default: `False`)
- `y_reference` (optional, default: `None`)

**Methods:**

##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


---

#### deepxde.callbacks.OperatorPredictor

```python
class deepxde.callbacks.OperatorPredictor(x, op, period=1, filename=None, precision=2)
```

Bases: Callback

**Parameters:**

- `x`: The input data.
- `op`: The operator with inputs (x, y).
- `period` (optional, default: `1`)
- `filename` (optional, default: `None`)
- `precision` (optional, default: `2`)

**Methods:**

##### `get_value`

```python
get_value()
```


##### `init`

```python
init()
```

Init after setting a model.


##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_predict_end`

```python
on_predict_end()
```

Called at the end of prediction.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


---

#### deepxde.callbacks.PDEPointResampler

```python
class deepxde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=False)
```

Bases: Callback

**Parameters:**

- `period` (optional, default: `100`): How often to resample the training points (default is 100 iterations).
- `pde_points` (optional, default: `True`): If True, resample the training points for PDE losses (default is
True).
- `bc_points` (optional, default: `False`): If True, resample the training points for BC losses (default is
False; only supported by PyTorch and PaddlePaddle backend currently).

**Methods:**

##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


---

#### deepxde.callbacks.Timer

```python
class deepxde.callbacks.Timer(available_time)
```

Bases: Callback

**Parameters:**

- `available_time`

**Methods:**

##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


---

#### deepxde.callbacks.VariableValue

```python
class deepxde.callbacks.VariableValue(var_list, period=1, filename=None, precision=2)
```

Bases: Callback

**Parameters:**

- `var_list`: A TensorFlow Variable
or a list of TensorFlow Variable.
- `period` (optional, default: `1`)
- `filename` (optional, default: `None`)
- `precision` (optional, default: `2`)

**Methods:**

##### `get_value`

```python
get_value()
```

Return the variable values.


##### `on_epoch_end`

```python
on_epoch_end()
```

Called at the end of every epoch.


##### `on_train_begin`

```python
on_train_begin()
```

Called at the beginning of model training.


##### `on_train_end`

```python
on_train_end()
```

Called at the end of model training.


---

#### deepxde.model.LossHistory

```python
class deepxde.model.LossHistory
```

Bases: object

**Methods:**

##### `append`

```python
append(step, loss_train, loss_test, metrics_test)
```

**Parameters:**

- `step`
- `loss_train`
- `loss_test`
- `metrics_test`


---

#### deepxde.model.Model

```python
class deepxde.model.Model(data, net)
```

Bases: object

**Parameters:**

- `data`: deepxde.data.Data instance.
- `net`: deepxde.nn.NN instance.

**Methods:**

##### `compile`

```python
compile(optimizer, lr=None, loss='MSE', metrics=None, decay=None, loss_weights=None, external_trainable_variables=None, verbose=1)
```

Configures the model for training.

**Parameters:**

- `optimizer`: String name of an optimizer, or a backend optimizer class
instance.
- `lr` (optional, default: `None`)
- `loss` (optional, default: `'MSE'`): If the same loss is used for all errors, then loss is a String name
of a loss function or a loss function. If different errors use
different losses, then loss is a list whose size is equal to the
number of errors.
- `metrics` (optional, default: `None`): List of metrics to be evaluated by the model during training.
- `decay` (optional, default: `None`)
- `loss_weights` (optional, default: `None`): A list specifying scalar coefficients (Python floats) to
weight the loss contributions. The loss value that will be minimized by
the model will then be the weighted sum of all individual losses,
weighted by the loss_weights coefficients.
- `external_trainable_variables` (optional, default: `None`): A trainable dde.Variable object or a list
of trainable dde.Variable objects. The unknown parameters in the
physics systems that need to be recovered. Regularization will not be
applied to these variables. If the backend is tensorflow.compat.v1,
external_trainable_variables is ignored, and all trainable dde.Variable
objects are automatically collected.
- `verbose` (optional, default: `1`)


##### `predict`

```python
predict(x, operator=None, callbacks=None)
```

Generates predictions for the input samples. If operator is None, returns the network output, otherwise returns the output of the operator.

**Parameters:**

- `x`: The network inputs. A Numpy array or a tuple of Numpy arrays.
- `operator` (optional, default: `None`): A function takes arguments (inputs, outputs) or (inputs,
outputs, auxiliary_variables) and outputs a tensor. inputs and
outputs are the network input and output tensors, respectively.
auxiliary_variables is the output of auxiliary_var_function(x)
in dde.data.PDE. operator is typically chosen as the PDE (used to
define dde.data.PDE) to predict the PDE residual.
- `callbacks` (optional, default: `None`): List of dde.callbacks.Callback instances. List of callbacks
to apply during prediction.


##### `print_model`

```python
print_model()
```

Prints all trainable variables.


##### `restore`

```python
restore(save_path, device=None, verbose=0)
```

Restore all variables from a disk file.

**Parameters:**

- `save_path`
- `device` (optional, default: `None`)
- `verbose` (optional, default: `0`)


##### `save`

```python
save(save_path, protocol='backend', verbose=0)
```

Saves all variables to a disk file.

**Parameters:**

- `save_path`
- `protocol` (optional, default: `'backend'`)
- `verbose` (optional, default: `0`)


##### `state_dict`

```python
state_dict()
```

Returns a dictionary containing all variables.


##### `train`

```python
train(iterations=None, batch_size=None, display_every=1000, disregard_previous_best=False, callbacks=None, model_restore_path=None, model_save_path=None, epochs=None, verbose=1)
```

Trains the model.

**Parameters:**

- `iterations` (optional, default: `None`)
- `batch_size` (optional, default: `None`): Integer, tuple, or None.

If you solve PDEs via dde.data.PDE or dde.data.TimePDE, do not use batch_size, and instead use
dde.callbacks.PDEPointResampler,
see an example.
For DeepONet in the format of Cartesian product, if batch_size is an Integer,
then it is the batch size for the branch input; if you want to also use mini-batch for the trunk net input,
set batch_size as a tuple, where the fist number is the batch size for the branch net input
and the second number is the batch size for the trunk net input.
- `display_every` (optional, default: `1000`)
- `disregard_previous_best` (optional, default: `False`): If True, disregard the previous saved best
model.
- `callbacks` (optional, default: `None`): List of dde.callbacks.Callback instances. List of callbacks
to apply during training.
- `model_restore_path` (optional, default: `None`)
- `model_save_path` (optional, default: `None`)
- `epochs` (optional, default: `None`)
- `verbose` (optional, default: `1`)


---

#### deepxde.model.TrainState

```python
class deepxde.model.TrainState
```

Bases: object

**Methods:**

##### `disregard_best`

```python
disregard_best()
```


##### `set_data_test`

```python
set_data_test(X_test, y_test, test_aux_vars=None)
```

**Parameters:**

- `X_test`
- `y_test`
- `test_aux_vars` (optional, default: `None`)


##### `set_data_train`

```python
set_data_train(X_train, y_train, train_aux_vars=None)
```

**Parameters:**

- `X_train`
- `y_train`
- `train_aux_vars` (optional, default: `None`)


##### `update_best`

```python
update_best()
```


---


---

## deepxde.backend

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.backend.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.backend.html)


---

## deepxde.data

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.data.html)

### Classes

#### deepxde.data.constraint.Constraint

```python
class deepxde.data.constraint.Constraint(constraint, train_x, test_x)
```

Bases: Data

**Parameters:**

- `constraint`
- `train_x`
- `test_x`

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.data.Data

```python
class deepxde.data.data.Data
```

Bases: ABC

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `losses_test`

```python
losses_test(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for test dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `losses_train`

```python
losses_train(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for training dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
abstractmethod test()
```

Return a test dataset.


##### `train_next_batch`

```python
abstractmethod train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.data.Tuple

```python
class deepxde.data.data.Tuple(train_x, train_y, test_x, test_y)
```

Bases: Data

**Parameters:**

- `train_x`
- `train_y`
- `test_x`
- `test_y`

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.dataset.DataSet

```python
class deepxde.data.dataset.DataSet(X_train=None, y_train=None, X_test=None, y_test=None, fname_train=None, fname_test=None, col_x=None, col_y=None, standardize=False)
```

Bases: Data

**Parameters:**

- `X_train` (optional, default: `None`)
- `y_train` (optional, default: `None`)
- `X_test` (optional, default: `None`)
- `y_test` (optional, default: `None`)
- `fname_train` (optional, default: `None`)
- `fname_test` (optional, default: `None`)
- `col_x` (optional, default: `None`): List of integers.
- `col_y` (optional, default: `None`): List of integers.
- `standardize` (optional, default: `False`)

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


##### `transform_inputs`

```python
transform_inputs(x)
```

**Parameters:**

- `x`


---

#### deepxde.data.fpde.FPDE

```python
class deepxde.data.fpde.FPDE(geometry, fpde, alpha, bcs, resolution, meshtype='dynamic', num_domain=0, num_boundary=0, train_distribution='Hammersley', anchors=None, solution=None, num_test=None)
```

Bases: PDE

**Parameters:**

- `geometry`
- `fpde`
- `alpha`
- `bcs`
- `resolution`
- `meshtype` (optional, default: `'dynamic'`)
- `num_domain` (optional, default: `0`)
- `num_boundary` (optional, default: `0`)
- `train_distribution` (optional, default: `'Hammersley'`)
- `anchors` (optional, default: `None`)
- `solution` (optional, default: `None`)
- `num_test` (optional, default: `None`)

**Methods:**

##### `get_int_matrix`

```python
get_int_matrix(training)
```

**Parameters:**

- `training`


##### `losses_test`

```python
losses_test(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for test dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `losses_train`

```python
losses_train(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for training dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `test_points`

```python
test_points()
```


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.fpde.Scheme

```python
class deepxde.data.fpde.Scheme(meshtype, resolution)
```

Bases: object

**Parameters:**

- `meshtype`
- `resolution`: A list of integer. The first number is the number of quadrature points in the first direction, …,
and the last number is the GL parameter.

---

#### deepxde.data.fpde.TimeFPDE

```python
class deepxde.data.fpde.TimeFPDE(geometryxtime, fpde, alpha, ic_bcs, resolution, meshtype='dynamic', num_domain=0, num_boundary=0, num_initial=0, train_distribution='Hammersley', anchors=None, solution=None, num_test=None)
```

Bases: FPDE

**Parameters:**

- `geometryxtime`
- `fpde`
- `alpha`
- `ic_bcs`
- `resolution`
- `meshtype` (optional, default: `'dynamic'`)
- `num_domain` (optional, default: `0`)
- `num_boundary` (optional, default: `0`)
- `num_initial` (optional, default: `0`)
- `train_distribution` (optional, default: `'Hammersley'`)
- `anchors` (optional, default: `None`)
- `solution` (optional, default: `None`)
- `num_test` (optional, default: `None`)

**Methods:**

##### `get_int_matrix`

```python
get_int_matrix(training)
```

**Parameters:**

- `training`


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


##### `train_points`

```python
train_points()
```


---

#### deepxde.data.func_constraint.FuncConstraint

```python
class deepxde.data.func_constraint.FuncConstraint(geom, constraint, func, num_train, anchors, num_test, dist_train='uniform')
```

Bases: Data

**Parameters:**

- `geom`
- `constraint`
- `func`
- `num_train`
- `anchors`
- `num_test`
- `dist_train` (optional, default: `'uniform'`)

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.function.Function

```python
class deepxde.data.function.Function(geometry, function, num_train, num_test, train_distribution='uniform', online=False)
```

Bases: Data

**Parameters:**

- `geometry`: The domain of the function. Instance of Geometry.
- `function`: The function to be approximated. A callable function takes a NumPy array as the input and returns the
a NumPy array of corresponding function values.
- `num_train`
- `num_test`
- `train_distribution` (optional, default: `'uniform'`)
- `online` (optional, default: `False`)

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.function_spaces.Chebyshev

```python
class deepxde.data.function_spaces.Chebyshev(N=100, M=1)
```

Bases: FunctionSpace

**Parameters:**

- `N` (optional, default: `100`)
- `M` (optional, default: `1`)

**Methods:**

##### `eval_batch`

```python
eval_batch(features, xs)
```

Evaluate a list of functions at a list of points.

**Parameters:**

- `features`: A NumPy array of shape (n_functions, n_features). A list of the
feature vectors of the functions to be evaluated.
- `xs`: A NumPy array of shape (n_points, dim). A list of points to be
evaluated.


##### `eval_one`

```python
eval_one(feature, x)
```

Evaluate the function at one point.

**Parameters:**

- `feature`: The feature vector of the function to be evaluated.
- `x`: The point to be evaluated.


##### `random`

```python
random(size)
```

Generate feature vectors of random functions.

**Parameters:**

- `size`


---

#### deepxde.data.function_spaces.FunctionSpace

```python
class deepxde.data.function_spaces.FunctionSpace
```

Bases: ABC

**Methods:**

##### `eval_batch`

```python
abstractmethod eval_batch(features, xs)
```

Evaluate a list of functions at a list of points.

**Parameters:**

- `features`: A NumPy array of shape (n_functions, n_features). A list of the
feature vectors of the functions to be evaluated.
- `xs`: A NumPy array of shape (n_points, dim). A list of points to be
evaluated.


##### `eval_one`

```python
abstractmethod eval_one(feature, x)
```

Evaluate the function at one point.

**Parameters:**

- `feature`: The feature vector of the function to be evaluated.
- `x`: The point to be evaluated.


##### `random`

```python
abstractmethod random(size)
```

Generate feature vectors of random functions.

**Parameters:**

- `size`


---

#### deepxde.data.function_spaces.GRF

```python
class deepxde.data.function_spaces.GRF(T=1, kernel='RBF', length_scale=1, N=1000, interp='cubic')
```

Bases: FunctionSpace

**Parameters:**

- `T` (optional, default: `1`)
- `kernel` (optional, default: `'RBF'`)
- `length_scale` (optional, default: `1`)
- `N` (optional, default: `1000`)
- `interp` (optional, default: `'cubic'`)

**Methods:**

##### `eval_batch`

```python
eval_batch(features, xs)
```

Evaluate a list of functions at a list of points.

**Parameters:**

- `features`: A NumPy array of shape (n_functions, n_features). A list of the
feature vectors of the functions to be evaluated.
- `xs`: A NumPy array of shape (n_points, dim). A list of points to be
evaluated.


##### `eval_one`

```python
eval_one(feature, x)
```

Evaluate the function at one point.

**Parameters:**

- `feature`: The feature vector of the function to be evaluated.
- `x`: The point to be evaluated.


##### `random`

```python
random(size)
```

Generate feature vectors of random functions.

**Parameters:**

- `size`


---

#### deepxde.data.function_spaces.GRF2D

```python
class deepxde.data.function_spaces.GRF2D(kernel='RBF', length_scale=1, N=100, interp='splinef2d')
```

Bases: FunctionSpace

**Parameters:**

- `kernel` (optional, default: `'RBF'`)
- `length_scale` (optional, default: `1`)
- `N` (optional, default: `100`)
- `interp` (optional, default: `'splinef2d'`)

**Methods:**

##### `eval_batch`

```python
eval_batch(features, xs)
```

Evaluate a list of functions at a list of points.

**Parameters:**

- `features`: A NumPy array of shape (n_functions, n_features). A list of the
feature vectors of the functions to be evaluated.
- `xs`: A NumPy array of shape (n_points, dim). A list of points to be
evaluated.


##### `eval_one`

```python
eval_one(feature, x)
```

Evaluate the function at one point.

**Parameters:**

- `feature`: The feature vector of the function to be evaluated.
- `x`: The point to be evaluated.


##### `random`

```python
random(size)
```

Generate feature vectors of random functions.

**Parameters:**

- `size`


---

#### deepxde.data.function_spaces.GRF_KL

```python
class deepxde.data.function_spaces.GRF_KL(T=1, kernel='RBF', length_scale=1, num_eig=10, N=100, interp='cubic')
```

Bases: FunctionSpace

**Parameters:**

- `T` (optional, default: `1`)
- `kernel` (optional, default: `'RBF'`)
- `length_scale` (optional, default: `1`)
- `num_eig` (optional, default: `10`)
- `N` (optional, default: `100`)
- `interp` (optional, default: `'cubic'`)

**Methods:**

##### `bases`

```python
bases(sensors)
```

Evaluate the eigenfunctions at a list of points sensors.

**Parameters:**

- `sensors`


##### `eval_batch`

```python
eval_batch(features, xs)
```

Evaluate a list of functions at a list of points.

**Parameters:**

- `features`: A NumPy array of shape (n_functions, n_features). A list of the
feature vectors of the functions to be evaluated.
- `xs`: A NumPy array of shape (n_points, dim). A list of points to be
evaluated.


##### `eval_one`

```python
eval_one(feature, x)
```

Evaluate the function at one point.

**Parameters:**

- `feature`: The feature vector of the function to be evaluated.
- `x`: The point to be evaluated.


##### `random`

```python
random(size)
```

Generate feature vectors of random functions.

**Parameters:**

- `size`


---

#### deepxde.data.function_spaces.PowerSeries

```python
class deepxde.data.function_spaces.PowerSeries(N=100, M=1)
```

Bases: FunctionSpace

**Parameters:**

- `N` (optional, default: `100`)
- `M` (optional, default: `1`)

**Methods:**

##### `eval_batch`

```python
eval_batch(features, xs)
```

Evaluate a list of functions at a list of points.

**Parameters:**

- `features`: A NumPy array of shape (n_functions, n_features). A list of the
feature vectors of the functions to be evaluated.
- `xs`: A NumPy array of shape (n_points, dim). A list of points to be
evaluated.


##### `eval_one`

```python
eval_one(feature, x)
```

Evaluate the function at one point.

**Parameters:**

- `feature`: The feature vector of the function to be evaluated.
- `x`: The point to be evaluated.


##### `random`

```python
random(size)
```

Generate feature vectors of random functions.

**Parameters:**

- `size`


---

#### deepxde.data.ide.IDE

```python
class deepxde.data.ide.IDE(geometry, ide, bcs, quad_deg, kernel=None, num_domain=0, num_boundary=0, train_distribution='Hammersley', anchors=None, solution=None, num_test=None)
```

Bases: PDE

**Parameters:**

- `geometry`
- `ide`
- `bcs`
- `quad_deg`
- `kernel` (optional, default: `None`)
- `num_domain` (optional, default: `0`)
- `num_boundary` (optional, default: `0`)
- `train_distribution` (optional, default: `'Hammersley'`)
- `anchors` (optional, default: `None`)
- `solution` (optional, default: `None`)
- `num_test` (optional, default: `None`)

**Methods:**

##### `get_int_matrix`

```python
get_int_matrix(training)
```

**Parameters:**

- `training`


##### `losses_test`

```python
losses_test(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for test dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `losses_train`

```python
losses_train(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for training dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `quad_points`

```python
quad_points(X)
```

**Parameters:**

- `X`


##### `test`

```python
test()
```

Return a test dataset.


##### `test_points`

```python
test_points()
```


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.mf.MfDataSet

```python
class deepxde.data.mf.MfDataSet(X_lo_train=None, X_hi_train=None, y_lo_train=None, y_hi_train=None, X_hi_test=None, y_hi_test=None, fname_lo_train=None, fname_hi_train=None, fname_hi_test=None, col_x=None, col_y=None, standardize=False)
```

Bases: Data

**Parameters:**

- `X_lo_train` (optional, default: `None`)
- `X_hi_train` (optional, default: `None`)
- `y_lo_train` (optional, default: `None`)
- `y_hi_train` (optional, default: `None`)
- `X_hi_test` (optional, default: `None`)
- `y_hi_test` (optional, default: `None`)
- `fname_lo_train` (optional, default: `None`)
- `fname_hi_train` (optional, default: `None`)
- `fname_hi_test` (optional, default: `None`)
- `col_x` (optional, default: `None`): List of integers.
- `col_y` (optional, default: `None`): List of integers.
- `standardize` (optional, default: `False`)

**Methods:**

##### `losses_test`

```python
losses_test(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for test dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `losses_train`

```python
losses_train(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for training dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.mf.MfFunc

```python
class deepxde.data.mf.MfFunc(geom, func_lo, func_hi, num_lo, num_hi, num_test, dist_train='uniform')
```

Bases: Data

**Parameters:**

- `geom`
- `func_lo`
- `func_hi`
- `num_lo`
- `num_hi`
- `num_test`
- `dist_train` (optional, default: `'uniform'`)

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.pde.PDE

```python
class deepxde.data.pde.PDE(geometry, pde, bcs, num_domain=0, num_boundary=0, train_distribution='Hammersley', anchors=None, exclusions=None, solution=None, num_test=None, auxiliary_var_function=None)
```

Bases: Data

**Parameters:**

- `geometry`: Instance of Geometry.
- `pde`: A global PDE or a list of PDEs. None if no global PDE.
- `bcs`: A boundary condition or a list of boundary conditions. Use [] if no
boundary condition.
- `num_domain` (optional, default: `0`)
- `num_boundary` (optional, default: `0`)
- `train_distribution` (optional, default: `'Hammersley'`)
- `anchors` (optional, default: `None`): A Numpy array of training points, in addition to the num_domain and
num_boundary sampled points.
- `exclusions` (optional, default: `None`): A Numpy array of points to be excluded for training.
- `solution` (optional, default: `None`): The reference solution.
- `num_test` (optional, default: `None`): The number of points sampled inside the domain for testing PDE loss.
The testing points for BCs/ICs are the same set of points used for training.
If None, then the training points will be used for testing.
- `auxiliary_var_function` (optional, default: `None`): A function that inputs train_x or test_x and outputs
auxiliary variables.

**Methods:**

##### `add_anchors`

```python
add_anchors(anchors)
```

Add new points for training PDE losses.

**Parameters:**

- `anchors`


##### `bc_points`

```python
bc_points()
```


##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `replace_with_anchors`

```python
replace_with_anchors(anchors)
```

Replace the current PDE training points with anchors.

**Parameters:**

- `anchors`


##### `resample_train_points`

```python
resample_train_points(pde_points=True, bc_points=True)
```

Resample the training points for PDE and/or BC.

**Parameters:**

- `pde_points` (optional, default: `True`)
- `bc_points` (optional, default: `True`)


##### `test`

```python
test()
```

Return a test dataset.


##### `test_points`

```python
test_points()
```


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


##### `train_points`

```python
train_points()
```


---

#### deepxde.data.pde.TimePDE

```python
class deepxde.data.pde.TimePDE(geometryxtime, pde, ic_bcs, num_domain=0, num_boundary=0, num_initial=0, train_distribution='Hammersley', anchors=None, exclusions=None, solution=None, num_test=None, auxiliary_var_function=None)
```

Bases: PDE

**Parameters:**

- `geometryxtime`
- `pde`
- `ic_bcs`
- `num_domain` (optional, default: `0`)
- `num_boundary` (optional, default: `0`)
- `num_initial` (optional, default: `0`)
- `train_distribution` (optional, default: `'Hammersley'`)
- `anchors` (optional, default: `None`)
- `exclusions` (optional, default: `None`)
- `solution` (optional, default: `None`)
- `num_test` (optional, default: `None`)
- `auxiliary_var_function` (optional, default: `None`)

**Methods:**

##### `train_points`

```python
train_points()
```


---

#### deepxde.data.pde_operator.PDEOperator

```python
class deepxde.data.pde_operator.PDEOperator(pde, function_space, evaluation_points, num_function, function_variables=None, num_test=None)
```

Bases: Data

**Parameters:**

- `pde`: Instance of dde.data.PDE or dde.data.TimePDE.
- `function_space`: Instance of dde.data.FunctionSpace.
- `evaluation_points`: A NumPy array of shape (n_points, dim). Discretize the input
function sampled from function_space using pointwise evaluations at a set
of points as the input of the branch net.
- `num_function`
- `function_variables` (optional, default: `None`): None or a list of integers. The functions in the
function_space may not have the same domain as the PDE. For example, the
PDE is defined on a spatio-temporal domain (x, t), but the function is
IC, which is only a function of x. In this case, we need to specify the
variables of the function by function_variables=[0], where 0 indicates
the first variable x. If None, then we assume the domains of the
function and the PDE are the same.
- `num_test` (optional, default: `None`): The number of functions for testing PDE loss. The testing functions
for BCs/ICs are the same functions used for training. If None, then the
training functions will be used for testing.

**Methods:**

##### `bc_inputs`

```python
bc_inputs(func_feats, func_vals)
```

**Parameters:**

- `func_feats`
- `func_vals`


##### `gen_inputs`

```python
gen_inputs(func_feats, func_vals, points)
```

**Parameters:**

- `func_feats`
- `func_vals`
- `points`


##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `resample_train_points`

```python
resample_train_points(pde_points=True, bc_points=True)
```

Resample the training points for the operator.

**Parameters:**

- `pde_points` (optional, default: `True`)
- `bc_points` (optional, default: `True`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.pde_operator.PDEOperatorCartesianProd

```python
class deepxde.data.pde_operator.PDEOperatorCartesianProd(pde, function_space, evaluation_points, num_function, function_variables=None, num_test=None, batch_size=None)
```

Bases: Data

**Parameters:**

- `pde`: Instance of dde.data.PDE or dde.data.TimePDE.
- `function_space`: Instance of dde.data.FunctionSpace.
- `evaluation_points`: A NumPy array of shape (n_points, dim). Discretize the input
function sampled from function_space using pointwise evaluations at a set
of points as the input of the branch net.
- `num_function`
- `function_variables` (optional, default: `None`): None or a list of integers. The functions in the
function_space may not have the same domain as the PDE. For example, the
PDE is defined on a spatio-temporal domain (x, t), but the function is
IC, which is only a function of x. In this case, we need to specify the
variables of the function by function_variables=[0], where 0 indicates
the first variable x. If None, then we assume the domains of the
function and the PDE are the same.
- `num_test` (optional, default: `None`): The number of functions for testing PDE loss. The testing functions
for BCs/ICs are the same functions used for training. If None, then the
training functions will be used for testing.
- `batch_size` (optional, default: `None`): Integer or None.

**Methods:**

##### `losses_test`

```python
losses_test(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for test dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `losses_train`

```python
losses_train(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses for training dataset, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.quadruple.Quadruple

```python
class deepxde.data.quadruple.Quadruple(X_train, y_train, X_test, y_test)
```

Bases: Data

**Parameters:**

- `X_train`: A tuple of three NumPy arrays.
- `y_train`: A NumPy array.
- `X_test`
- `y_test`

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.quadruple.QuadrupleCartesianProd

```python
class deepxde.data.quadruple.QuadrupleCartesianProd(X_train, y_train, X_test, y_test)
```

Bases: Data

**Parameters:**

- `X_train`: A tuple of three NumPy arrays. The first element has the shape (N1,
dim1), the second element has the shape (N1, dim2), and the third
element has the shape (N2, dim3).
- `y_train`: A NumPy array of shape (N1, N2).
- `X_test`
- `y_test`

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.sampler.BatchSampler

```python
class deepxde.data.sampler.BatchSampler(num_samples, shuffle=True)
```

Bases: object

**Parameters:**

- `num_samples`
- `shuffle` (optional, default: `True`)

**Methods:**

##### `get_next`

```python
get_next(batch_size)
```

Returns the indices of the next batch.

**Parameters:**

- `batch_size`


---

#### deepxde.data.triple.Triple

```python
class deepxde.data.triple.Triple(X_train, y_train, X_test, y_test)
```

Bases: Data

**Parameters:**

- `X_train`: A tuple of two NumPy arrays.
- `y_train`: A NumPy array.
- `X_test`
- `y_test`

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---

#### deepxde.data.triple.TripleCartesianProd

```python
class deepxde.data.triple.TripleCartesianProd(X_train, y_train, X_test, y_test)
```

Bases: Data

**Parameters:**

- `X_train`: A tuple of two NumPy arrays. The first element has the shape (N1,
dim1), and the second element has the shape (N2, dim2).
- `y_train`: A NumPy array of shape (N1, N2).
- `X_test`
- `y_test`

**Methods:**

##### `losses`

```python
losses(targets, outputs, loss_fn, inputs, model, aux=None)
```

Return a list of losses, i.e., constraints.

**Parameters:**

- `targets`
- `outputs`
- `loss_fn`
- `inputs`
- `model`
- `aux` (optional, default: `None`)


##### `test`

```python
test()
```

Return a test dataset.


##### `train_next_batch`

```python
train_next_batch(batch_size=None)
```

Return a training dataset of the size batch_size.

**Parameters:**

- `batch_size` (optional, default: `None`)


---


---

## deepxde.geometry

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.geometry.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.geometry.html)

### Classes

#### deepxde.geometry.csg.CSGDifference

```python
class deepxde.geometry.csg.CSGDifference(geom1, geom2)
```

Bases: Geometry

**Parameters:**

- `geom1`
- `geom2`

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component)
```

Compute the periodic image of x for periodic boundary condition.

**Parameters:**

- `x`
- `component`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


---

#### deepxde.geometry.csg.CSGIntersection

```python
class deepxde.geometry.csg.CSGIntersection(geom1, geom2)
```

Bases: Geometry

**Parameters:**

- `geom1`
- `geom2`

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component)
```

Compute the periodic image of x for periodic boundary condition.

**Parameters:**

- `x`
- `component`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


---

#### deepxde.geometry.csg.CSGUnion

```python
class deepxde.geometry.csg.CSGUnion(geom1, geom2)
```

Bases: Geometry

**Parameters:**

- `geom1`
- `geom2`

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component)
```

Compute the periodic image of x for periodic boundary condition.

**Parameters:**

- `x`
- `component`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


---

#### deepxde.geometry.geometry.Geometry

```python
class deepxde.geometry.geometry.Geometry(dim, bbox, diam)
```

Bases: ABC

**Parameters:**

- `dim`
- `bbox`
- `diam`

**Methods:**

##### `background_points`

```python
background_points(x, dirn, dist2npt, shift)
```

**Parameters:**

- `x`
- `dirn`
- `dist2npt`
- `shift`


##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+')
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)


##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `difference`

```python
difference(other)
```

CSG Difference.

**Parameters:**

- `other`


##### `distance2boundary`

```python
distance2boundary(x, dirn)
```

**Parameters:**

- `x`
- `dirn`


##### `inside`

```python
abstractmethod inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `intersection`

```python
intersection(other)
```

CSG Intersection.

**Parameters:**

- `other`


##### `mindist2boundary`

```python
mindist2boundary(x)
```

**Parameters:**

- `x`


##### `on_boundary`

```python
abstractmethod on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component)
```

Compute the periodic image of x for periodic boundary condition.

**Parameters:**

- `x`
- `component`


##### `random_boundary_points`

```python
abstractmethod random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
abstractmethod random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


##### `uniform_points`

```python
uniform_points(n, boundary=True)
```

Compute the equispaced point locations in the geometry.

**Parameters:**

- `n`
- `boundary` (optional, default: `True`)


##### `union`

```python
union(other)
```

CSG Union.

**Parameters:**

- `other`


---

#### deepxde.geometry.geometry_1d.Interval

```python
class deepxde.geometry.geometry_1d.Interval(l, r)
```

Bases: Geometry

**Parameters:**

- `l`
- `r`

**Methods:**

##### `background_points`

```python
background_points(x, dirn, dist2npt, shift)
```

dirn – -1 (left), or 1 (right), or 0 (both direction).

**Parameters:**

- `x`
- `dirn`: -1 (left), or 1 (right), or 0 (both direction).
- `dist2npt`: A function which converts distance to the number of extra
points (not including x).
- `shift`: The number of shift.


##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+', where: None | Literal['left', 'right'] = None)
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)
- `where: None | Literal['left'`
- `'right']` (optional, default: `None`)


##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `distance2boundary`

```python
distance2boundary(x, dirn)
```

**Parameters:**

- `x`
- `dirn`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `log_uniform_points`

```python
log_uniform_points(n, boundary=True)
```

**Parameters:**

- `n`
- `boundary` (optional, default: `True`)


##### `mindist2boundary`

```python
mindist2boundary(x)
```

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component=0)
```

Compute the periodic image of x for periodic boundary condition.

**Parameters:**

- `x`
- `component` (optional, default: `0`)


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


##### `uniform_points`

```python
uniform_points(n, boundary=True)
```

Compute the equispaced point locations in the geometry.

**Parameters:**

- `n`
- `boundary` (optional, default: `True`)


---

#### deepxde.geometry.geometry_2d.Disk

```python
class deepxde.geometry.geometry_2d.Disk(center, radius)
```

Bases: Hypersphere

**Parameters:**

- `center`
- `radius`

**Methods:**

##### `background_points`

```python
background_points(x, dirn, dist2npt, shift)
```

**Parameters:**

- `x`
- `dirn`
- `dist2npt`
- `shift`


##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `distance2boundary`

```python
distance2boundary(x, dirn)
```

**Parameters:**

- `x`
- `dirn`


##### `distance2boundary_unitdirn`

```python
distance2boundary_unitdirn(x, dirn)
```

**Parameters:**

- `x`
- `dirn`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `mindist2boundary`

```python
mindist2boundary(x)
```

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_2d.Ellipse

```python
class deepxde.geometry.geometry_2d.Ellipse(center, semimajor, semiminor, angle=0)
```

Bases: Geometry

**Parameters:**

- `center`: Center of the ellipse.
- `semimajor`: Semimajor of the ellipse.
- `semiminor`: Semiminor of the ellipse.
- `angle` (optional, default: `0`): Rotation angle of the ellipse. A positive angle rotates the ellipse
clockwise about the center and a negative angle rotates the ellipse
counterclockwise about the center.

**Methods:**

##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+')
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_2d.Polygon

```python
class deepxde.geometry.geometry_2d.Polygon(vertices)
```

Bases: Geometry

**Parameters:**

- `vertices`

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_2d.Rectangle

```python
class deepxde.geometry.geometry_2d.Rectangle(xmin, xmax)
```

Bases: Hypercube

**Parameters:**

- `xmin`: Coordinate of bottom left corner.
- `xmax`: Coordinate of top right corner.

**Methods:**

##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+', where: None | Literal['left', 'right', 'bottom', 'top'] = None, inside: bool = True)
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)
- `where: None | Literal['left'`
- `'right'`
- `'bottom'`
- `'top']` (optional, default: `None`)
- `inside: bool` (optional, default: `True`)


##### `is_valid`

```python
static is_valid(vertices)
```

Check if the geometry is a Rectangle.

**Parameters:**

- `vertices`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_2d.StarShaped

```python
class deepxde.geometry.geometry_2d.StarShaped(center, radius, coeffs_cos, coeffs_sin)
```

Bases: Geometry

**Parameters:**

- `center`: Center of the domain.
- `radius`: 0th-order term of the parametrization (r_0).
- `coeffs_cos`: i-th order coefficients for the i-th cos term (a_i).
- `coeffs_sin`: i-th order coefficients for the i-th sin term (b_i).

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_2d.Triangle

```python
class deepxde.geometry.geometry_2d.Triangle(x1, x2, x3)
```

Bases: Geometry

**Parameters:**

- `x1`
- `x2`
- `x3`

**Methods:**

##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+', where: None | Literal['x1-x2', 'x1-x3', 'x2-x3'] = None)
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)
- `where: None | Literal['x1-x2'`
- `'x1-x3'`
- `'x2-x3']` (optional, default: `None`)


##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_3d.Cuboid

```python
class deepxde.geometry.geometry_3d.Cuboid(xmin, xmax)
```

Bases: Hypercube

**Parameters:**

- `xmin`: Coordinate of bottom left corner.
- `xmax`: Coordinate of top right corner.

**Methods:**

##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+', where: None | Literal['back', 'front', 'left', 'right', 'bottom', 'top'] = None, inside: bool = True)
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)
- `where: None | Literal['back'`
- `'front'`
- `'left'`
- `'right'`
- `'bottom'`
- `'top']` (optional, default: `None`)
- `inside: bool` (optional, default: `True`)


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


---

#### deepxde.geometry.geometry_3d.Sphere

```python
class deepxde.geometry.geometry_3d.Sphere(center, radius)
```

Bases: Hypersphere

**Parameters:**

- `center`: Center of the sphere.
- `radius`: Radius of the sphere.

---

#### deepxde.geometry.geometry_nd.Hypercube

```python
class deepxde.geometry.geometry_nd.Hypercube(xmin, xmax)
```

Bases: Geometry

**Parameters:**

- `xmin`
- `xmax`

**Methods:**

##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0', where: None = None, inside: bool = True)
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0'`)
- `where: None` (optional, default: `None`)
- `inside: bool` (optional, default: `True`)


##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component)
```

Compute the periodic image of x for periodic boundary condition.

**Parameters:**

- `x`
- `component`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Compute the equispaced point locations on the boundary.

**Parameters:**

- `n`


##### `uniform_points`

```python
uniform_points(n, boundary=True)
```

Compute the equispaced point locations in the geometry.

**Parameters:**

- `n`
- `boundary` (optional, default: `True`)


---

#### deepxde.geometry.geometry_nd.Hypersphere

```python
class deepxde.geometry.geometry_nd.Hypersphere(center, radius)
```

Bases: Geometry

**Parameters:**

- `center`
- `radius`

**Methods:**

##### `background_points`

```python
background_points(x, dirn, dist2npt, shift)
```

**Parameters:**

- `x`
- `dirn`
- `dist2npt`
- `shift`


##### `boundary_constraint_factor`

```python
boundary_constraint_factor(x, smoothness: Literal['C0', 'C0+', 'Cinf'] = 'C0+')
```

Compute the hard constraint factor at x for the boundary.

**Parameters:**

- `x`: A 2D array of shape (n, dim), where n is the number of points and
dim is the dimension of the geometry. Note that x should be a tensor type
of backend (e.g., tf.Tensor or torch.Tensor), not a numpy array.
- `smoothness: Literal['C0'`
- `'C0+'`
- `'Cinf']` (optional, default: `'C0+'`)


##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `distance2boundary`

```python
distance2boundary(x, dirn)
```

**Parameters:**

- `x`
- `dirn`


##### `distance2boundary_unitdirn`

```python
distance2boundary_unitdirn(x, dirn)
```

**Parameters:**

- `x`
- `dirn`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `mindist2boundary`

```python
mindist2boundary(x)
```

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


---

#### deepxde.geometry.pointcloud.PointCloud

```python
class deepxde.geometry.pointcloud.PointCloud(points, boundary_points=None, boundary_normals=None)
```

Bases: Geometry

**Parameters:**

- `points`: A 2-D NumPy array. If boundary_points is not provided, points can
include points both inside the geometry or on the boundary; if boundary_points
is provided, points includes only points inside the geometry.
- `boundary_points` (optional, default: `None`): A 2-D NumPy array.
- `boundary_normals` (optional, default: `None`): A 2-D NumPy array.

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

Compute the unit normal at x for Neumann or Robin boundary conditions.

**Parameters:**

- `x`


##### `inside`

```python
inside(x)
```

Check if x is inside the geometry (including the boundary).

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

Check if x is on the geometry boundary.

**Parameters:**

- `x`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

Compute the random point locations on the boundary.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

Compute the random point locations in the geometry.

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


---

#### deepxde.geometry.timedomain.GeometryXTime

```python
class deepxde.geometry.timedomain.GeometryXTime(geometry, timedomain)
```

Bases: object

**Parameters:**

- `geometry`
- `timedomain`

**Methods:**

##### `boundary_normal`

```python
boundary_normal(x)
```

**Parameters:**

- `x`


##### `on_boundary`

```python
on_boundary(x)
```

**Parameters:**

- `x`


##### `on_initial`

```python
on_initial(x)
```

**Parameters:**

- `x`


##### `periodic_point`

```python
periodic_point(x, component)
```

**Parameters:**

- `x`
- `component`


##### `random_boundary_points`

```python
random_boundary_points(n, random='pseudo')
```

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_initial_points`

```python
random_initial_points(n, random='pseudo')
```

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `random_points`

```python
random_points(n, random='pseudo')
```

**Parameters:**

- `n`
- `random` (optional, default: `'pseudo'`)


##### `uniform_boundary_points`

```python
uniform_boundary_points(n)
```

Uniform boundary points on the spatio-temporal domain.

**Parameters:**

- `n`


##### `uniform_initial_points`

```python
uniform_initial_points(n)
```

**Parameters:**

- `n`


##### `uniform_points`

```python
uniform_points(n, boundary=True)
```

Uniform points on the spatio-temporal domain.

**Parameters:**

- `n`
- `boundary` (optional, default: `True`)


---

#### deepxde.geometry.timedomain.TimeDomain

```python
class deepxde.geometry.timedomain.TimeDomain(t0, t1)
```

Bases: Interval

**Parameters:**

- `t0`
- `t1`

**Methods:**

##### `on_initial`

```python
on_initial(t)
```

**Parameters:**

- `t`


---


---

## deepxde.gradients

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.gradients.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.gradients.html)


---

## deepxde.icbc

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.icbc.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.icbc.html)

### Classes

#### deepxde.icbc.boundary_conditions.BC

```python
class deepxde.icbc.boundary_conditions.BC(geom, on_boundary, component)
```

Bases: ABC

**Parameters:**

- `geom`: A deepxde.geometry.Geometry instance.
- `on_boundary`: A function: (x, Geometry.on_boundary(x)) -> True/False.
- `component`: The output component satisfying this BC.

**Methods:**

##### `collocation_points`

```python
collocation_points(X)
```

**Parameters:**

- `X`


##### `error`

```python
abstractmethod error(X, inputs, outputs, beg, end, aux_var=None)
```

Returns the loss.

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


##### `filter`

```python
filter(X)
```

**Parameters:**

- `X`


##### `normal_derivative`

```python
normal_derivative(X, inputs, outputs, beg, end)
```

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`


---

#### deepxde.icbc.boundary_conditions.DirichletBC

```python
class deepxde.icbc.boundary_conditions.DirichletBC(geom, func, on_boundary, component=0)
```

Bases: BC

**Parameters:**

- `geom`
- `func`
- `on_boundary`
- `component` (optional, default: `0`)

**Methods:**

##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

Returns the loss.

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.Interface2DBC

```python
class deepxde.icbc.boundary_conditions.Interface2DBC(geom, func, on_boundary1, on_boundary2, direction='normal')
```

Bases: object

**Parameters:**

- `geom`: a dde.geometry.Rectangle or dde.geometry.Polygon instance.
- `func`: the target discontinuity between edges, evaluated on the first edge,
e.g., func=lambda x: 0 means no discontinuity is wanted.
- `on_boundary1`: First edge func. (x, Geometry.on_boundary(x)) -> True/False.
- `on_boundary2`: Second edge func. (x, Geometry.on_boundary(x)) -> True/False.
- `direction` (optional, default: `'normal'`)

**Methods:**

##### `collocation_points`

```python
collocation_points(X)
```

**Parameters:**

- `X`


##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.NeumannBC

```python
class deepxde.icbc.boundary_conditions.NeumannBC(geom, func, on_boundary, component=0)
```

Bases: BC

**Parameters:**

- `geom`
- `func`
- `on_boundary`
- `component` (optional, default: `0`)

**Methods:**

##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

Returns the loss.

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.OperatorBC

```python
class deepxde.icbc.boundary_conditions.OperatorBC(geom, func, on_boundary)
```

Bases: BC

**Parameters:**

- `geom`: Geometry.
- `func`: A function takes arguments (inputs, outputs, X)
and outputs a tensor of size N x 1, where N is the length of inputs.
inputs and outputs are the network input and output tensors,
respectively; X are the NumPy array of the inputs.
- `on_boundary`: (x, Geometry.on_boundary(x)) -> True/False.

**Methods:**

##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

Returns the loss.

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.PeriodicBC

```python
class deepxde.icbc.boundary_conditions.PeriodicBC(geom, component_x, on_boundary, derivative_order=0, component=0)
```

Bases: BC

**Parameters:**

- `geom`
- `component_x`
- `on_boundary`
- `derivative_order` (optional, default: `0`)
- `component` (optional, default: `0`)

**Methods:**

##### `collocation_points`

```python
collocation_points(X)
```

**Parameters:**

- `X`


##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

Returns the loss.

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.PointSetBC

```python
class deepxde.icbc.boundary_conditions.PointSetBC(points, values, component=0, batch_size=None, shuffle=True)
```

Bases: object

**Parameters:**

- `points`: An array of points where the corresponding target values are known and
used for training.
- `values`: A scalar or a 2D-array of values that gives the exact solution of the problem.
- `component` (optional, default: `0`): Integer or a list of integers. The output components satisfying this BC.
List of integers only supported for the backend PyTorch.
- `batch_size` (optional, default: `None`): The number of points per minibatch, or None to return all points.
This is only supported for the backend PyTorch and PaddlePaddle.
Note, If you want to use batch size here, you should also set callback
‘dde.callbacks.PDEPointResampler(bc_points=True)’ in training.
- `shuffle` (optional, default: `True`): Randomize the order on each pass through the data when batching.

**Methods:**

##### `collocation_points`

```python
collocation_points(X)
```

**Parameters:**

- `X`


##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.PointSetOperatorBC

```python
class deepxde.icbc.boundary_conditions.PointSetOperatorBC(points, values, func, batch_size=None, shuffle=True)
```

Bases: object

**Parameters:**

- `points`: An array of points where the corresponding target values are
known and used for training.
- `values`: An array of values which output of function should fulfill.
- `func`: A function takes arguments (inputs, outputs, X)
and outputs a tensor of size N x 1, where N is the length of
inputs. inputs and outputs are the network input and output
tensors, respectively; X are the NumPy array of the inputs.
- `batch_size` (optional, default: `None`): The number of points per minibatch, or None to return all points.
This is only supported for the backend PyTorch and PaddlePaddle.
Note, If you want to use batch size here, you should also set callback
‘dde.callbacks.PDEPointResampler(bc_points=True)’ in training.
- `shuffle` (optional, default: `True`): Randomize the order on each pass through the data when batching.

**Methods:**

##### `collocation_points`

```python
collocation_points(X)
```

**Parameters:**

- `X`


##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.boundary_conditions.RobinBC

```python
class deepxde.icbc.boundary_conditions.RobinBC(geom, func, on_boundary, component=0)
```

Bases: BC

**Parameters:**

- `geom`
- `func`
- `on_boundary`
- `component` (optional, default: `0`)

**Methods:**

##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

Returns the loss.

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


---

#### deepxde.icbc.initial_conditions.IC

```python
class deepxde.icbc.initial_conditions.IC(geom, func, on_initial, component=0)
```

Bases: object

**Parameters:**

- `geom`
- `func`
- `on_initial`
- `component` (optional, default: `0`)

**Methods:**

##### `collocation_points`

```python
collocation_points(X)
```

**Parameters:**

- `X`


##### `error`

```python
error(X, inputs, outputs, beg, end, aux_var=None)
```

**Parameters:**

- `X`
- `inputs`
- `outputs`
- `beg`
- `end`
- `aux_var` (optional, default: `None`)


##### `filter`

```python
filter(X)
```

**Parameters:**

- `X`


---


---

## deepxde.nn

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.html)

### Classes

#### deepxde.nn.initializers.VarianceScalingStacked

```python
class deepxde.nn.initializers.VarianceScalingStacked(scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None)
```

Bases: object

**Parameters:**

- `scale` (optional, default: `1.0`): Scaling factor (positive float).
- `mode` (optional, default: `'fan_in'`): One of “fan_in”, “fan_out”, “fan_avg”.
- `distribution` (optional, default: `'truncated_normal'`): Random distribution to use. One of “normal”, “uniform”.
- `seed` (optional, default: `None`): A Python integer. Used to create random seeds. See
tf.set_random_seed
for behavior.

---


---

## deepxde.nn.jax

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.jax.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.jax.html)

### Classes

#### deepxde.nn.jax.fnn.FNN

```python
class deepxde.nn.jax.fnn.FNN(layer_sizes: ~typing.Any, activation: ~typing.Any, kernel_initializer: ~typing.Any, regularization: ~typing.Any = None, params: ~typing.Any = None, _input_transform: ~typing.Callable = None, _output_transform: ~typing.Callable = None, parent: ~flax.linen.module.Module | ~flax.core.scope.Scope | ~flax.linen.module._Sentinel | None = <flax.linen.module._Sentinel object>, name: str | None = None)
```

Bases: NN

**Parameters:**

- `layer_sizes: ~typing.Any`
- `activation: ~typing.Any`
- `kernel_initializer: ~typing.Any`
- `regularization: ~typing.Any` (optional, default: `None`)
- `params: ~typing.Any` (optional, default: `None`)
- `_input_transform: ~typing.Callable` (optional, default: `None`)
- `_output_transform: ~typing.Callable` (optional, default: `None`)
- `parent: ~flax.linen.module.Module | ~flax.core.scope.Scope | ~flax.linen.module._Sentinel | None` (optional, default: `<flax.linen.module._Sentinel object>`)
- `name: str | None` (optional, default: `None`)

**Methods:**

##### `setup`

```python
setup()
```

Initializes a Module lazily (similar to a lazy __init__).


---

#### deepxde.nn.jax.fnn.PFNN

```python
class deepxde.nn.jax.fnn.PFNN(layer_sizes: ~typing.Any, activation: ~typing.Any, kernel_initializer: ~typing.Any, regularization: ~typing.Any = None, params: ~typing.Any = None, _input_transform: ~typing.Callable = None, _output_transform: ~typing.Callable = None, parent: ~flax.linen.module.Module | ~flax.core.scope.Scope | ~flax.linen.module._Sentinel | None = <flax.linen.module._Sentinel object>, name: str | None = None)
```

Bases: NN

**Parameters:**

- `layer_sizes: ~typing.Any`
- `activation: ~typing.Any`
- `kernel_initializer: ~typing.Any`
- `regularization: ~typing.Any` (optional, default: `None`)
- `params: ~typing.Any` (optional, default: `None`)
- `_input_transform: ~typing.Callable` (optional, default: `None`)
- `_output_transform: ~typing.Callable` (optional, default: `None`)
- `parent: ~flax.linen.module.Module | ~flax.core.scope.Scope | ~flax.linen.module._Sentinel | None` (optional, default: `<flax.linen.module._Sentinel object>`)
- `name: str | None` (optional, default: `None`)

**Methods:**

##### `setup`

```python
setup()
```

Initializes a Module lazily (similar to a lazy __init__).


---

#### deepxde.nn.jax.nn.NN

```python
class deepxde.nn.jax.nn.NN(parent: ~flax.linen.module.Module | ~flax.core.scope.Scope | ~flax.linen.module._Sentinel | None = <flax.linen.module._Sentinel object>, name: str | None = None)
```

Bases: Module

**Parameters:**

- `parent: ~flax.linen.module.Module | ~flax.core.scope.Scope | ~flax.linen.module._Sentinel | None` (optional, default: `<flax.linen.module._Sentinel object>`)
- `name: str | None` (optional, default: `None`)

**Methods:**

##### `apply_feature_transform`

```python
apply_feature_transform(transform)
```

Compute the features by appling a transform to the network inputs, i.e., features = transform(inputs). Then, outputs = network(features).

**Parameters:**

- `transform`


##### `apply_output_transform`

```python
apply_output_transform(transform)
```

Apply a transform to the network outputs, i.e., outputs = transform(inputs, outputs).

**Parameters:**

- `transform`


---


---

## deepxde.nn.paddle

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.paddle.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.paddle.html)

### Classes

#### deepxde.nn.paddle.deeponet.DeepONet

```python
class deepxde.nn.paddle.deeponet.DeepONet(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, use_bias=True)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch`: A list of integers as the width of a fully connected
network, or (dim, f) where dim is the input dimension and f is a
network function. The width of the last layer in the branch and trunk net
should be equal.
- `layer_sizes_trunk`
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `use_bias` (optional, default: `True`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Defines the computation performed at every call. Should be overridden by all subclasses.

**Parameters:**

- `inputs`


---

#### deepxde.nn.paddle.deeponet.DeepONetCartesianProd

```python
class deepxde.nn.paddle.deeponet.DeepONetCartesianProd(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, num_outputs=1, multi_output_strategy=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch`: A list of integers as the width of a fully connected network,
or (dim, f) where dim is the input dimension and f is a network
function. The width of the last layer in the branch and trunk net
should be the same for all strategies except “split_branch” and “split_trunk”.
- `layer_sizes_trunk`
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `num_outputs` (optional, default: `1`)
- `multi_output_strategy` (optional, default: `None`)

**Methods:**

##### `build_branch_net`

```python
build_branch_net(layer_sizes_branch)
```

**Parameters:**

- `layer_sizes_branch`


##### `build_trunk_net`

```python
build_trunk_net(layer_sizes_trunk)
```

**Parameters:**

- `layer_sizes_trunk`


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `forward`

```python
forward(inputs)
```

Defines the computation performed at every call. Should be overridden by all subclasses.

**Parameters:**

- `inputs`


##### `merge_branch_trunk`

```python
merge_branch_trunk(x_func, x_loc, index)
```

**Parameters:**

- `x_func`
- `x_loc`
- `index`


---

#### deepxde.nn.paddle.fnn.FNN

```python
class deepxde.nn.paddle.fnn.FNN(layer_sizes, activation, kernel_initializer, regularization=None, dropout_rate=0)
```

Bases: NN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Defines the computation performed at every call. Should be overridden by all subclasses.

**Parameters:**

- `inputs`


---

#### deepxde.nn.paddle.fnn.PFNN

```python
class deepxde.nn.paddle.fnn.PFNN(layer_sizes, activation, kernel_initializer, regularization=None)
```

Bases: NN

**Parameters:**

- `layer_sizes`: A nested list that defines the architecture of the neural network
(how the layers are connected). If layer_sizes[i] is an int, it represents
one layer shared by all the outputs; if layer_sizes[i] is a list, it
represents len(layer_sizes[i]) sub-layers, each of which is exclusively
used by one output. Note that len(layer_sizes[i]) should equal the number
of outputs. Every number specifies the number of neurons in that layer.
- `activation`: A string represent activation used in fully-connected net.
- `kernel_initializer`: Initializer for the kernel weights matrix.
- `regularization` (optional, default: `None`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Defines the computation performed at every call. Should be overridden by all subclasses.

**Parameters:**

- `inputs`


---

#### deepxde.nn.paddle.msffn.MsFFN

```python
class deepxde.nn.paddle.msffn.MsFFN(layer_sizes, activation, kernel_initializer, sigmas, dropout_rate=0)
```

Bases: NN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `sigmas`
- `dropout_rate` (optional, default: `0`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Defines the computation performed at every call. Should be overridden by all subclasses.

**Parameters:**

- `inputs`


---

#### deepxde.nn.paddle.msffn.STMsFFN

```python
class deepxde.nn.paddle.msffn.STMsFFN(layer_sizes, activation, kernel_initializer, sigmas_x, sigmas_t, dropout_rate=0)
```

Bases: MsFFN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `sigmas_x`
- `sigmas_t`
- `dropout_rate` (optional, default: `0`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Defines the computation performed at every call. Should be overridden by all subclasses.

**Parameters:**

- `inputs`


---

#### deepxde.nn.paddle.nn.NN

```python
class deepxde.nn.paddle.nn.NN
```

Bases: Layer

**Methods:**

##### `apply_feature_transform`

```python
apply_feature_transform(transform)
```

Compute the features by appling a transform to the network inputs, i.e., features = transform(inputs). Then, outputs = network(features).

**Parameters:**

- `transform`


##### `apply_output_transform`

```python
apply_output_transform(transform)
```

Apply a transform to the network outputs, i.e., outputs = transform(inputs, outputs).

**Parameters:**

- `transform`


---


---

## deepxde.nn.pytorch

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.pytorch.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.pytorch.html)

### Classes

#### deepxde.nn.pytorch.deeponet.DeepONet

```python
class deepxde.nn.pytorch.deeponet.DeepONet(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, num_outputs=1, multi_output_strategy=None, regularization=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch`: A list of integers as the width of a fully connected network,
or (dim, f) where dim is the input dimension and f is a network
function. The width of the last layer in the branch and trunk net
should be the same for all strategies except “split_branch” and “split_trunk”.
- `layer_sizes_trunk`
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `num_outputs` (optional, default: `1`)
- `multi_output_strategy` (optional, default: `None`)
- `regularization` (optional, default: `None`)

**Methods:**

##### `build_branch_net`

```python
build_branch_net(layer_sizes_branch)
```

**Parameters:**

- `layer_sizes_branch`


##### `build_trunk_net`

```python
build_trunk_net(layer_sizes_trunk)
```

**Parameters:**

- `layer_sizes_trunk`


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


##### `merge_branch_trunk`

```python
merge_branch_trunk(x_func, x_loc, index)
```

**Parameters:**

- `x_func`
- `x_loc`
- `index`


---

#### deepxde.nn.pytorch.deeponet.DeepONetCartesianProd

```python
class deepxde.nn.pytorch.deeponet.DeepONetCartesianProd(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, num_outputs=1, multi_output_strategy=None, regularization=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch`: A list of integers as the width of a fully connected network,
or (dim, f) where dim is the input dimension and f is a network
function. The width of the last layer in the branch and trunk net
should be the same for all strategies except “split_branch” and “split_trunk”.
- `layer_sizes_trunk`
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `num_outputs` (optional, default: `1`)
- `multi_output_strategy` (optional, default: `None`)
- `regularization` (optional, default: `None`)

**Methods:**

##### `build_branch_net`

```python
build_branch_net(layer_sizes_branch)
```

**Parameters:**

- `layer_sizes_branch`


##### `build_trunk_net`

```python
build_trunk_net(layer_sizes_trunk)
```

**Parameters:**

- `layer_sizes_trunk`


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


##### `merge_branch_trunk`

```python
merge_branch_trunk(x_func, x_loc, index)
```

**Parameters:**

- `x_func`
- `x_loc`
- `index`


---

#### deepxde.nn.pytorch.deeponet.PODDeepONet

```python
class deepxde.nn.pytorch.deeponet.PODDeepONet(pod_basis, layer_sizes_branch, activation, kernel_initializer, layer_sizes_trunk=None, regularization=None)
```

Bases: NN

**Parameters:**

- `pod_basis`: POD basis used in the trunk net.
- `layer_sizes_branch`: A list of integers as the width of a fully connected network,
or (dim, f) where dim is the input dimension and f is a network
function. The width of the last layer in the branch and trunk net should be
equal.
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `layer_sizes_trunk` (optional, default: `None`)
- `regularization` (optional, default: `None`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


---

#### deepxde.nn.pytorch.fnn.FNN

```python
class deepxde.nn.pytorch.fnn.FNN(layer_sizes, activation, kernel_initializer, regularization=None)
```

Bases: NN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


---

#### deepxde.nn.pytorch.fnn.PFNN

```python
class deepxde.nn.pytorch.fnn.PFNN(layer_sizes, activation, kernel_initializer)
```

Bases: NN

**Parameters:**

- `layer_sizes`: A nested list that defines the architecture of the neural network
(how the layers are connected). If layer_sizes[i] is an int, it represents
one layer shared by all the outputs; if layer_sizes[i] is a list, it
represents len(layer_sizes[i]) sub-layers, each of which is exclusively
used by one output. Every list in layer_sizes must have the same length
(= number of subnetworks). If the last element of layer_sizes is an int
preceded by a list, it must be equal to the number of subnetworks: all
subnetworks have an output size of 1 and are then concatenated. If the last
element is a list, it specifies the output size for each subnetwork before
concatenation.
- `activation`: Activation function.
- `kernel_initializer`: Initializer for the kernel weights.

**Methods:**

##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


---

#### deepxde.nn.pytorch.mionet.MIONetCartesianProd

```python
class deepxde.nn.pytorch.mionet.MIONetCartesianProd(layer_sizes_branch1, layer_sizes_branch2, layer_sizes_trunk, activation, kernel_initializer, regularization=None, trunk_last_activation=False, merge_operation='mul', layer_sizes_merger=None, output_merge_operation='mul', layer_sizes_output_merger=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch1`
- `layer_sizes_branch2`
- `layer_sizes_trunk`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `trunk_last_activation` (optional, default: `False`)
- `merge_operation` (optional, default: `'mul'`)
- `layer_sizes_merger` (optional, default: `None`)
- `output_merge_operation` (optional, default: `'mul'`)
- `layer_sizes_output_merger` (optional, default: `None`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


---

#### deepxde.nn.pytorch.mionet.PODMIONet

```python
class deepxde.nn.pytorch.mionet.PODMIONet(pod_basis, layer_sizes_branch1, layer_sizes_branch2, activation, kernel_initializer, layer_sizes_trunk=None, regularization=None, trunk_last_activation=False, merge_operation='mul', layer_sizes_merger=None)
```

Bases: NN

**Parameters:**

- `pod_basis`
- `layer_sizes_branch1`
- `layer_sizes_branch2`
- `activation`
- `kernel_initializer`
- `layer_sizes_trunk` (optional, default: `None`)
- `regularization` (optional, default: `None`)
- `trunk_last_activation` (optional, default: `False`)
- `merge_operation` (optional, default: `'mul'`)
- `layer_sizes_merger` (optional, default: `None`)

**Methods:**

##### `forward`

```python
forward(inputs)
```

Define the computation performed at every call.

**Parameters:**

- `inputs`


---

#### deepxde.nn.pytorch.nn.NN

```python
class deepxde.nn.pytorch.nn.NN
```

Bases: Module

**Methods:**

##### `apply_feature_transform`

```python
apply_feature_transform(transform)
```

Compute the features by appling a transform to the network inputs, i.e., features = transform(inputs). Then, outputs = network(features).

**Parameters:**

- `transform`


##### `apply_output_transform`

```python
apply_output_transform(transform)
```

Apply a transform to the network outputs, i.e., outputs = transform(inputs, outputs).

**Parameters:**

- `transform`


##### `num_trainable_parameters`

```python
num_trainable_parameters()
```

Evaluate the number of trainable parameters for the NN.


---


---

## deepxde.nn.tensorflow

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.tensorflow.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.tensorflow.html)

### Classes

#### deepxde.nn.tensorflow.deeponet.DeepONet

```python
class deepxde.nn.tensorflow.deeponet.DeepONet(*args, **kwargs)
```

Bases: NN

**Parameters:**

- `*args`
- `**kwargs`

**Methods:**

##### `build_branch_net`

```python
build_branch_net(layer_sizes_branch)
```

**Parameters:**

- `layer_sizes_branch`


##### `build_trunk_net`

```python
build_trunk_net(layer_sizes_trunk)
```

**Parameters:**

- `layer_sizes_trunk`


##### `call`

```python
call(inputs, training=False)
```

**Parameters:**

- `inputs`
- `training` (optional, default: `False`)


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `merge_branch_trunk`

```python
merge_branch_trunk(x_func, x_loc, index)
```

**Parameters:**

- `x_func`
- `x_loc`
- `index`


---

#### deepxde.nn.tensorflow.deeponet.DeepONetCartesianProd

```python
class deepxde.nn.tensorflow.deeponet.DeepONetCartesianProd(*args, **kwargs)
```

Bases: NN

**Parameters:**

- `*args`
- `**kwargs`

**Methods:**

##### `build_branch_net`

```python
build_branch_net(layer_sizes_branch)
```

**Parameters:**

- `layer_sizes_branch`


##### `build_trunk_net`

```python
build_trunk_net(layer_sizes_trunk)
```

**Parameters:**

- `layer_sizes_trunk`


##### `call`

```python
call(inputs, training=False)
```

**Parameters:**

- `inputs`
- `training` (optional, default: `False`)


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `merge_branch_trunk`

```python
merge_branch_trunk(x_func, x_loc, index)
```

**Parameters:**

- `x_func`
- `x_loc`
- `index`


---

#### deepxde.nn.tensorflow.deeponet.PODDeepONet

```python
class deepxde.nn.tensorflow.deeponet.PODDeepONet(*args, **kwargs)
```

Bases: NN

**Parameters:**

- `*args`
- `**kwargs`

**Methods:**

##### `call`

```python
call(inputs, training=False)
```

**Parameters:**

- `inputs`
- `training` (optional, default: `False`)


---

#### deepxde.nn.tensorflow.fnn.FNN

```python
class deepxde.nn.tensorflow.fnn.FNN(*args, **kwargs)
```

Bases: NN

**Parameters:**

- `*args`
- `**kwargs`

**Methods:**

##### `call`

```python
call(inputs, training=False)
```

**Parameters:**

- `inputs`
- `training` (optional, default: `False`)


---

#### deepxde.nn.tensorflow.fnn.PFNN

```python
class deepxde.nn.tensorflow.fnn.PFNN(*args, **kwargs)
```

Bases: NN

**Parameters:**

- `*args`
- `**kwargs`

**Methods:**

##### `call`

```python
call(inputs, training=False)
```

**Parameters:**

- `inputs`
- `training` (optional, default: `False`)


---

#### deepxde.nn.tensorflow.nn.NN

```python
class deepxde.nn.tensorflow.nn.NN(*args, **kwargs)
```

Bases: Model

**Parameters:**

- `*args`
- `**kwargs`

**Methods:**

##### `apply_feature_transform`

```python
apply_feature_transform(transform)
```

Compute the features by appling a transform to the network inputs, i.e., features = transform(inputs). Then, outputs = network(features).

**Parameters:**

- `transform`


##### `apply_output_transform`

```python
apply_output_transform(transform)
```

Apply a transform to the network outputs, i.e., outputs = transform(inputs, outputs).

**Parameters:**

- `transform`


##### `num_trainable_parameters`

```python
num_trainable_parameters()
```

Evaluate the number of trainable parameters for the NN.


---


---

## deepxde.nn.tensorflow_compat_v1

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.tensorflow_compat_v1.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.tensorflow_compat_v1.html)

### Classes

#### deepxde.nn.tensorflow_compat_v1.deeponet.DeepONet

```python
class deepxde.nn.tensorflow_compat_v1.deeponet.DeepONet(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, regularization=None, dropout_rate=0, use_bias=True, stacked=False, trainable_branch=True, trainable_trunk=True, num_outputs=1, multi_output_strategy=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch`: A list of integers as the width of a fully connected
network, or (dim, f) where dim is the input dimension and f is a
network function. The width of the last layer in the branch and trunk net
should be the same for all strategies except “split_branch” and “split_trunk”.
- `layer_sizes_trunk`
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`): If dropout_rate is a float between 0 and 1, then the
same rate is used in both trunk and branch nets. If dropout_rate
is a dict, then the trunk net uses the rate dropout_rate[“trunk”],
and the branch net uses dropout_rate[“branch”]. Both dropout_rate[“trunk”]
and dropout_rate[“branch”] should be float or lists of float.
The list length should match the length of layer_sizes_trunk - 1 for the
trunk net and layer_sizes_branch - 2 for the branch net.
- `use_bias` (optional, default: `True`)
- `stacked` (optional, default: `False`)
- `trainable_branch` (optional, default: `True`): Boolean.
- `trainable_trunk` (optional, default: `True`): Boolean or a list of booleans.
- `num_outputs` (optional, default: `1`)
- `multi_output_strategy` (optional, default: `None`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


##### `build_branch_net`

```python
build_branch_net()
```


##### `build_trunk_net`

```python
build_trunk_net()
```


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `merge_branch_trunk`

```python
merge_branch_trunk(branch, trunk)
```

**Parameters:**

- `branch`
- `trunk`


---

#### deepxde.nn.tensorflow_compat_v1.deeponet.DeepONetCartesianProd

```python
class deepxde.nn.tensorflow_compat_v1.deeponet.DeepONetCartesianProd(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, regularization=None, dropout_rate=0, num_outputs=1, multi_output_strategy=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch`: A list of integers as the width of a fully connected network,
or (dim, f) where dim is the input dimension and f is a network
function. The width of the last layer in the branch and trunk net
should be the same for all strategies except “split_branch” and “split_trunk”.
- `layer_sizes_trunk`
- `activation`: If activation is a string, then the same activation is used in
both trunk and branch nets. If activation is a dict, then the trunk
net uses the activation activation[“trunk”], and the branch net uses
activation[“branch”].
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`): If dropout_rate is a float between 0 and 1, then the
same rate is used in both trunk and branch nets. If dropout_rate
is a dict, then the trunk net uses the rate dropout_rate[“trunk”],
and the branch net uses dropout_rate[“branch”]. Both dropout_rate[“trunk”]
and dropout_rate[“branch”] should be float or lists of float.
The list length should match the length of layer_sizes_trunk - 1 for the
trunk net and layer_sizes_branch - 2 for the branch net.
- `num_outputs` (optional, default: `1`)
- `multi_output_strategy` (optional, default: `None`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


##### `build_branch_net`

```python
build_branch_net()
```


##### `build_trunk_net`

```python
build_trunk_net()
```


##### `concatenate_outputs`

```python
static concatenate_outputs(ys)
```

**Parameters:**

- `ys`


##### `merge_branch_trunk`

```python
merge_branch_trunk(branch, trunk)
```

**Parameters:**

- `branch`
- `trunk`


---

#### deepxde.nn.tensorflow_compat_v1.fnn.FNN

```python
class deepxde.nn.tensorflow_compat_v1.fnn.FNN(layer_sizes, activation, kernel_initializer, regularization=None, dropout_rate=0, batch_normalization=None, layer_normalization=None, kernel_constraint=None, use_bias=True)
```

Bases: NN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`)
- `batch_normalization` (optional, default: `None`)
- `layer_normalization` (optional, default: `None`)
- `kernel_constraint` (optional, default: `None`)
- `use_bias` (optional, default: `True`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.fnn.PFNN

```python
class deepxde.nn.tensorflow_compat_v1.fnn.PFNN(layer_sizes, activation, kernel_initializer, regularization=None, dropout_rate=0, batch_normalization=None)
```

Bases: FNN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`)
- `batch_normalization` (optional, default: `None`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.mfnn.MfNN

```python
class deepxde.nn.tensorflow_compat_v1.mfnn.MfNN(layer_sizes_low_fidelity, layer_sizes_high_fidelity, activation, kernel_initializer, regularization=None, residue=False, trainable_low_fidelity=True, trainable_high_fidelity=True)
```

Bases: NN

**Parameters:**

- `layer_sizes_low_fidelity`
- `layer_sizes_high_fidelity`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)
- `residue` (optional, default: `False`)
- `trainable_low_fidelity` (optional, default: `True`)
- `trainable_high_fidelity` (optional, default: `True`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.mionet.MIONet

```python
class deepxde.nn.tensorflow_compat_v1.mionet.MIONet(layer_sizes_branch1, layer_sizes_branch2, layer_sizes_trunk, activation, kernel_initializer, regularization=None)
```

Bases: NN

**Parameters:**

- `layer_sizes_branch1`
- `layer_sizes_branch2`
- `layer_sizes_trunk`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.mionet.MIONetCartesianProd

```python
class deepxde.nn.tensorflow_compat_v1.mionet.MIONetCartesianProd(layer_sizes_branch1, layer_sizes_branch2, layer_sizes_trunk, activation, kernel_initializer, regularization=None)
```

Bases: MIONet

**Parameters:**

- `layer_sizes_branch1`
- `layer_sizes_branch2`
- `layer_sizes_trunk`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.msffn.MsFFN

```python
class deepxde.nn.tensorflow_compat_v1.msffn.MsFFN(layer_sizes, activation, kernel_initializer, sigmas, regularization=None, dropout_rate=0, batch_normalization=None, layer_normalization=None, kernel_constraint=None, use_bias=True)
```

Bases: FNN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `sigmas`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`)
- `batch_normalization` (optional, default: `None`)
- `layer_normalization` (optional, default: `None`)
- `kernel_constraint` (optional, default: `None`)
- `use_bias` (optional, default: `True`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.msffn.STMsFFN

```python
class deepxde.nn.tensorflow_compat_v1.msffn.STMsFFN(layer_sizes, activation, kernel_initializer, sigmas_x, sigmas_t, regularization=None, dropout_rate=0, batch_normalization=None, layer_normalization=None, kernel_constraint=None, use_bias=True)
```

Bases: MsFFN

**Parameters:**

- `layer_sizes`
- `activation`
- `kernel_initializer`
- `sigmas_x`
- `sigmas_t`
- `regularization` (optional, default: `None`)
- `dropout_rate` (optional, default: `0`)
- `batch_normalization` (optional, default: `None`)
- `layer_normalization` (optional, default: `None`)
- `kernel_constraint` (optional, default: `None`)
- `use_bias` (optional, default: `True`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---

#### deepxde.nn.tensorflow_compat_v1.nn.NN

```python
class deepxde.nn.tensorflow_compat_v1.nn.NN
```

Bases: object

**Methods:**

##### `apply_feature_transform`

```python
apply_feature_transform(transform)
```

Compute the features by appling a transform to the network inputs, i.e., features = transform(inputs). Then, outputs = network(features).

**Parameters:**

- `transform`


##### `apply_output_transform`

```python
apply_output_transform(transform)
```

Apply a transform to the network outputs, i.e., outputs = transform(inputs, outputs).

**Parameters:**

- `transform`


##### `build`

```python
build()
```

Construct the network.


##### `feed_dict`

```python
feed_dict(training, inputs, targets=None, auxiliary_vars=None)
```

Construct a feed_dict to feed values to TensorFlow placeholders.

**Parameters:**

- `training`
- `inputs`
- `targets` (optional, default: `None`)
- `auxiliary_vars` (optional, default: `None`)


##### `num_trainable_parameters`

```python
num_trainable_parameters()
```

Evaluate the number of trainable parameters for the NN.


---

#### deepxde.nn.tensorflow_compat_v1.resnet.ResNet

```python
class deepxde.nn.tensorflow_compat_v1.resnet.ResNet(input_size, output_size, num_neurons, num_blocks, activation, kernel_initializer, regularization=None)
```

Bases: NN

**Parameters:**

- `input_size`
- `output_size`
- `num_neurons`
- `num_blocks`
- `activation`
- `kernel_initializer`
- `regularization` (optional, default: `None`)

**Methods:**

##### `build`

```python
build()
```

Construct the network.


---


---

## deepxde.optimizers

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.optimizers.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.optimizers.html)


---

## deepxde.optimizers.pytorch

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.optimizers.pytorch.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.optimizers.pytorch.html)

### Classes

#### deepxde.optimizers.pytorch.nncg.NNCG

```python
class deepxde.optimizers.pytorch.nncg.NNCG(params, lr=1.0, rank=10, mu=0.0001, update_freq=20, chunk_size=1, cg_tol=1e-16, cg_max_iters=1000, line_search_fn=None, verbose=False)
```

Bases: Optimizer

**Parameters:**

- `params`
- `lr` (optional, default: `1.0`)
- `rank` (optional, default: `10`)
- `mu` (optional, default: `0.0001`)
- `update_freq` (optional, default: `20`)
- `chunk_size` (optional, default: `1`)
- `cg_tol` (optional, default: `1e-16`)
- `cg_max_iters` (optional, default: `1000`)
- `line_search_fn` (optional, default: `None`)
- `verbose` (optional, default: `False`)

**Methods:**

##### `step`

```python
step(closure)
```

Perform a single optimization step.

**Parameters:**

- `closure`


---


---

## deepxde.utils

**Documentation URL:** [https://deepxde.readthedocs.io/en/latest/modules/deepxde.utils.html](https://deepxde.readthedocs.io/en/latest/modules/deepxde.utils.html)

### Classes

#### deepxde.utils.external.PointSet

```python
class deepxde.utils.external.PointSet(points)
```

Bases: object

**Parameters:**

- `points`

**Methods:**

##### `inside`

```python
inside(x)
```

Returns True if x is in this set of points, otherwise, returns False.

**Parameters:**

- `x`


##### `values_to_func`

```python
values_to_func(values, default_value=0)
```

Convert the pairs of points and values to a callable function.

**Parameters:**

- `values`: A NumPy array of shape (N, dy). values[i] is the dy-dim
function value of the i-th point in this point set.
- `default_value` (optional, default: `0`)


---


---
