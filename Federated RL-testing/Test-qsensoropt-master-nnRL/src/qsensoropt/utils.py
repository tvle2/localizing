"""Module containing some utility functions
for training the control strategy and visualizing the
results.
"""
from typing import Tuple, TypedDict, Callable, \
    List

from tensorflow import GradientTape, function, zeros, transpose, zeros_like, print, Variable
from tensorflow import reshape, broadcast_to, \
    linspace, Tensor, dtypes
from tensorflow import range as tfrange
from tensorflow.linalg import svd, diag, matmul
from tensorflow.math import log, lgamma, abs, tanh
from tensorflow.math import sqrt as tfsqrt
from tensorflow.profiler.experimental import Trace, start, stop
from tensorflow.summary import trace_on, create_file_writer, \
    trace_export
from tensorflow.random import stateless_uniform, Generator
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from numpy import repeat, concatenate, savetxt, loadtxt, \
    linspace, inf
from numpy import sqrt as npsqrt
from numpy import floor as npfloor
from numpy import tanh as nptanh
from numpy import exp as npexp
import scipy.integrate as integrate
from pandas import DataFrame, Series
from os.path import join
from os import makedirs
from math import pi, floor
from datetime import datetime
from typing import Optional
from progressbar import progressbar

from .schedulers import InverseSqrtDecay


def normalize(
        input_tensor: Tensor,
        bounds: Tuple[float, float],
) -> Tensor:
    r"""Normalizes the entries of
    `input_tensor` in the interval `[-1, 1]`.
    The `input_tensor` entries must lay within the
    given tuple of upper and lower bounds.

    Parameters
    ----------
    input_tensor: Tensor
        `Tensor` of unnormalized values.
    bound: List[float, float]
        Min and max of the admissible entries
        of `input_tensor`.

    Returns
    -------
    Tensor:
        Version of `input_tensor` with entries
        normalized in the interval `[-1, 1]`.
        Each entry :math:`x` of the input becomes

        .. math::
            y = \frac{2(x-\text{bound}[0])}{\text{bound}[1]-
            \text{bound}[0]} - 1 \;

        in the returned `Tensor`, which has the same
        type and shape of `input_tensor`.
    """
    step_1 = (input_tensor-bounds[0])/(bounds[1]-bounds[0])
    return 2*step_1-1


def denormalize(
        input_tensor: Tensor,
        bounds: Tuple[float, float],
) -> Tensor:
    r"""Given `input_tensor` with values in `[-1, 1]`
    this function rescales it, so that its entries take
    values within the given tuple of extrema.

    Parameters
    ----------
    input_tensor: Tensor
        `Tensor` having entries normalized in `[-1, 1]`.
    bound: Tuple[float, float]
        Min and max of the admissible entries
        of the returned `Tensor`.

    Returns
    -------
    Tensor:
        Version of `input_tensor` normalized in
        the interval delimited by the new extrema.
        Each entry :math:`y` of `input_tensor` becomes

        .. math::
            x = \frac{y (\text{bound}[1]-
            \text{bound}[0])}{2}+\text{bounds}[0] \; .

        It is a `Tensor` with the same type and shape
        of `input_tensor`.
    """
    step_1 = (input_tensor+1)/2
    return (step_1*(bounds[1]-bounds[0]))+bounds[0]


def get_seed(
        random_generator: Generator,
) -> Tensor:
    """Generate a random seed from `random_generator`.

    Extracts uniformly a couple of integers from
    `random_generator` to be used as seed in the stateless
    functions of the :py:mod:`tensorflow.random` module.

    Parameters
    ----------
    random_generator: Generator
        Random number generator from the module
        :py:mod:`tensorflow.random`.

    Returns
    -------
    seed: Tensor
        `Tensor` of shape (2, ) and of type `int32`
        containing two random seeds.
    """
    return random_generator.uniform(
        [2, ], minval=0, maxval=dtypes.int32.max,
        dtype="int32", name="seed",
    )


def random_uniform(
        batchsize: int, prec: str,
        min_value: float, max_value: float,
        seed: Tensor,
):
    """Extracts `batchsize` random number of type `prec` between
    the `min_value` and the `max_value` from a uniform
    distribution.

    Parameters
    ----------
    batchsize: int
        Number of random values to be extracted.
    prec: str
        Type of the extracted numbers.
        It is typically `float32` or `float64`.
    min_values: float
        Lower extremum of the uniform distribution
        from which the values are extracted.
    max_values: float
        Upper extremum of the uniform distribution
        from which the values are extracted.
    seed: Tensor
        Seed of the random number generator used in this
        function. It is a `Tensor` of type `int32` and of shape
        (`2`, ). This is the kind of seed that is accepted by
        the stateless random
        function of the module :py:mod:`tensorflow.random`.
        It can be generated with the function
        :py:func:`~.utils.get_seed` from a
        :py:obj:`Generator` object.

    Returns
    -------
    Tensor:
        `Tensor` of shape (`batchsize`, ) and of
        type `prec` with uniformly extracted entries.
    """
    random_values = stateless_uniform(
        (batchsize, ), seed, maxval=1, dtype=prec,
        name="stateless_uniform",
    )
    return (max_value - min_value)*random_values + min_value


def store_input_control(
    simulation,
    idxN: int,
    data_dir: str,
    iterations: int,
    xla_compile: bool = True,
    rangen=Generator.from_seed(0xdeadd0d0),

):

    #@function(jit_compile=xla_compile)
    def simulate_nn(rangen):
        #
        true_values, history_input, history_control, \
            history_resources, _,_ = simulation.execute(idxN,
                rangen, deploy=True,
            )
        return true_values, history_input, \
            history_control, history_resources

    data = {"index": [], "values": [], "input": [],
            "control": [], "resources": []}
    for j in progressbar(range(iterations)):

        true_values, history_input, history_control, \
            history_resources = simulate_nn(
                rangen,
            )
        index_part = reshape(
            tfrange(j*simulation.bs, (j+1)*simulation.bs),
            (1, simulation.bs),
        )
        print('Sotr--', j, simulation.simpars.num_steps,index_part)
        index_part = reshape(repeat(
            index_part, repeats=[simulation.simpars.num_steps],
            axis=0,
        ), (simulation.bs*simulation.simpars.num_steps, 1)
        )
        true_values = reshape(broadcast_to(
            transpose(true_values, (1, 0, 2)),
            (simulation.simpars.num_steps, simulation.bs,
             simulation.phys_model.d),),
            (simulation.bs*simulation.simpars.num_steps,
             simulation.phys_model.d)
        )
        history_input = reshape(
            history_input,
            (simulation.bs*simulation.simpars.num_steps,
             simulation.input_size)
        )
        history_control = reshape(
            history_control,
            (simulation.bs*simulation.simpars.num_steps,
                simulation.phys_model.controls_size)
        )
        history_resources = reshape(
            history_resources,
            (simulation.bs*simulation.simpars.num_steps, )
        )
        data["index"].append(index_part.numpy())
        data["values"].append(true_values.numpy())
        data["input"].append(history_input.numpy())
        data["control"].append(history_control.numpy())
        data["resources"].append(history_resources.numpy())

    index_array = concatenate(data["index"], axis=0)
    values_array = concatenate(data["values"], axis=0)
    input_array = concatenate(data["input"], axis=0)
    control_array = concatenate(data["control"], axis=0)
    resources_array = concatenate(data["resources"], axis=0)

    data_array = concatenate(
        [index_array, values_array, input_array, control_array],
        axis=1,
    )

    columns = ['Estimation', ]
    for par in simulation.phys_model.params:
        columns.append(par.name)
    columns += simulation.input_name
    for contr in simulation.phys_model.controls:
        columns.append(contr.name)

    # Loading of the true values, the inputs and the controls
    # in a pandas data frame
    data_df = DataFrame(data=data_array, columns=columns)
    # This is needed to remove the blank lines in data_tf
    resources_df = DataFrame(
        data=resources_array, columns=['Resources', ],
    )
    data_df = data_df[resources_df['Resources'] != 0]
    data_df.to_csv(
        join(data_dir, str(simulation))+'_ext.csv',
        index=False,
        float_format='%.4e',
    )


class FitSpecifics(TypedDict):
    """This dictionary specifies the hyperparameters of the
    precision fit operated by the
    function :py:func:`~.utils.performance_evaluation`."""
    num_points: int
    """After the fit the neural network
    representing the relation between the resources
    and the average precision is evaluated
    on `num_points` resources values
    equally spaced in the
    interval [0, `max_resources`], with
    `max_resources` being the attribute
    of :py:obj:`~.SimulationParameters`."""
    batchsize: int
    """Batchsize of the training of the
    neural network to fit the
    Precision/Resources relation. The
    data cloud is divided in minibatches
    and each of them is used for a sigle
    iteration of the training loop."""
    epochs: int
    """The number of trainings on the
    same data."""
    direct_func: Callable
    r"""Callable object that takes in input
    the precision and the consumed resources
    and outputs a values :math:`x`
    that is of order one. In symbols

    .. math::
        f(\text{Resources}, \text{Precision})
        = x \sim \mathcal{O}(1) \; .

    This requires having some knowledge of the
    expected precision given the resources, which
    could be for example a Cramér-Rao bound
    on the precision."""
    inverse_func: Callable
    r"""Inverse of the function defined
    by `direct_func`, that is

    .. math::
        g(\text{Resources}, x) = \text{Precision} \; ."""


def performance_evaluation(
    simulation,
    idxN:int,
    iterations: int,
    data_dir: str,
    xla_compile: bool = True,
    precision_fit: Optional[FitSpecifics] = None,
    delta_resources: Optional[float] = None,
    y_label: str = 'Precision',
    rangen=Generator.from_seed(0xdeadd0d0),
):

    #@function(jit_compile=xla_compile)
    def simulate_nn(rangen):
        _, _, _, history_resources, history_precision,nnmse_loss = \
            simulation.execute(
                idxN,rangen, deploy=True,#9
            )
        return history_resources, history_precision,nnmse_loss

    precision_list = []
    resources_list = []
    nnmse_list = []
    for _ in progressbar(range(iterations)):
        history_resources, history_precision,nnmse_loss = simulate_nn(
            rangen,
        )
        history_resources = reshape(
            history_resources,
            (simulation.bs*simulation.simpars.num_steps, )
        )
        history_precision = reshape(
            history_precision,
            (simulation.bs*simulation.simpars.num_steps, )
        )
        resources_list.append(history_resources.numpy())
        precision_list.append(history_precision.numpy())
        nnmse_list.append(nnmse_loss)
    nnMse_loss = sum(nnmse_list) / len(nnmse_list)
    resources_array = concatenate(resources_list, axis=0)
    precision_array = concatenate(precision_list, axis=0)
    # Loading of the resources and the precisions in a Numpy array
    prec_df = DataFrame(
        {'Resources': resources_array, y_label: precision_array},
    )

    # The rows with zero resources are removed
    prec_df = prec_df[prec_df['Resources'] != 0]
    if delta_resources:

        bin_index_floor = npfloor(
            prec_df['Resources'].values/delta_resources,
        )
        # The number of resources associated to each bin
        # is the mean number of resources in that bin
        prec_df = prec_df.groupby(bin_index_floor).mean()
    if precision_fit:
        # Parametrization of the function that returns the precision
        # as a function of the resources
        prec_df = prec_df.sample(frac=1).reset_index(drop=True)

        # Create model for fitting the Precision/Resources relation
        model = standard_model()
        model.compile(
            optimizer=Adam(learning_rate=InverseSqrtDecay(1e-2)),
            loss='mean_squared_error',
        )
        max_res = simulation.simpars.max_resources

        model.fit(
            prec_df['Resources']/max_res,
            precision_fit['direct_func'](
                prec_df['Resources'], prec_df[y_label]),
            batch_size=precision_fit['batchsize'],
            epochs=precision_fit['epochs'],
            verbose=1,
        )
        # Generate the prediction for the precision
        x_axis = linspace(0, max_res, precision_fit['num_points']+1)[1:]
        y_axis = precision_fit['inverse_func'](
            x_axis, model(x_axis/max_res)[:, 0])
        prec_df = DataFrame(
            {'Resources': x_axis, y_label: y_axis},
        )

    prec_df['nnMse'] = nnMse_loss
    prec_df.to_csv(
        join(data_dir, str(simulation))+'_eval.csv',
        index=False,
        float_format='%.4e',
    )


def standard_model(
    input_size: int = 1,
    controls_size: int = 1,
    neurons_per_layer: int = 64,
    num_mid_layers: int = 5,
    prec: str = "float64",
    normalize_activation: bool = True,
    sigma_input: float = 0.33,
    last_layer_activation: str = "tanh",
) -> Model:

    # There should always be at least 2 intermediate layers.
    if num_mid_layers <= 1:
        raise ValueError("num_mid_layers must be > 1.")

    if normalize_activation:
        sigma_z_input = npsqrt(
            2*input_size/(input_size+neurons_per_layer))*sigma_input
        sigma_z_mid = sigma_input
        sigma_z_output = npsqrt(2*neurons_per_layer /
                                (controls_size+neurons_per_layer))*sigma_input

        C_input = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_input), -inf, inf,
        )[0]
        C_mid = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_mid), -inf, inf,
        )[0]
        C_output = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_output), -inf, inf)[0]

        input_norm, mid_norm, output_norm = sigma_input/npsqrt(C_input), \
            sigma_input/npsqrt(C_mid), sigma_input/npsqrt(C_output)

    else:
        input_norm, mid_norm, output_norm = 1.0, 1.0, 1.0

    layer_list = [
        Dense(neurons_per_layer,
              activation=lambda x: input_norm*tanh(x),
              dtype=prec, input_shape=(input_size, )),
    ]

    for _ in range(num_mid_layers-2):
        layer_list.append(
            Dense(neurons_per_layer,
                  activation=lambda x: mid_norm*tanh(x),
                  dtype=prec),
        )

    layer_list.append(
        Dense(neurons_per_layer,
              activation=lambda x: output_norm*tanh(x),
              dtype=prec),
    )

    layer_list.append(
        Dense(controls_size,
              activation=last_layer_activation,
              dtype=prec)
    )

    return Sequential(layer_list)


def npgauss_pdf(x, dev):
    """Logarithm of a batch of
    1D-Gaussian probability densities (compatible with
    Numpy)"""
    return 1/(npsqrt(2*pi)*dev)*npexp(-0.5*x**2/dev**2)


def loggauss(x, mean, dev):
    r"""Logarithm of a batch of
    1D-Gaussian probability densities.

    Parameters
    ----------
    x: Tensor
        Values extracted from the Gaussian distributions.
        Must have the same type and size of `mean` and `dev`.
    mean: Tensor
        Means of the Gaussian distributions.
        Must have the same type and size of `x` and `dev`.
    dev:
        Standard deviation of the Gaussian distributions.
        Must have the same type and size of `mean` and `x`.

    Returns
    -------
    Tensor:
        Logarithm of the probability densities for
        extracting the entries of `x` from the Gaussian
        distributions defined by `mean` and `dev`. It has
        the same shape and type of `x`, `mean`, and `dev`.

        Calling :math:`x`, :math:`\mu`, and :math:`\sigma`
        respectively an entry of the tensor `x`, `mean`, and `dev`,
        the corresponding entry of the returned tensor is

        .. math::
            -\log \left( \sqrt{2 \pi} \sigma \right) -
            \frac{(x-\mu)^2}{2 \sigma^2} \; .

    """
    return -log(npsqrt(2*pi)*dev)+(-0.5*((x-mean)/dev)**2)


def logpoisson(mean, k):
    r"""Logarithm of the probability densities
    of a batch of Poissonian distributions.

    Parameters
    ----------
    mean: Tensor
        Mean values defining the Poissonian distributions.
        Must have the same type and shape of `k`.
    k: Tensor
        Observed outcomes of the sampling from the
        Poissonian distributions.
        Must have the same type and shape of `k`.

    Returns
    -------
    Tensor:
        `Tensor` having the same type and shape
        of `mean` and `k`, whose entries are defined
        by

        .. math::
            k \log (\mu) - \mu - \log (k !) \; ,

        where :math:`k` and :math:`\mu` are respectively
        the entries of `k` and `mean`.
    """
    return k*log(mean) - mean - lgamma(k + 1)


def sqrt_hmatrix(matrix: Tensor) -> Tensor:
    """Square root of the absolute value of
    a symmetric (hermitian) matrix.

    The default matrix square root algorithm
    implemented in Tensorflow [7]_
    doesn't work for matrices with very small entries,
    this implementation does, and must be
    always preferred.

    Parameters
    ----------
    matrix: Tensor
        Batch of symmetric (hermitian) square matrices.

    Returns
    -------
    Tensor:
        Matrix square root of `matrix`.

    Examples
    --------
    ``A = constant([[1e-16, 1e-15], [1e-15, 1e-16]],
    dtype="float64", )``

    ``print(tf.sqrtm(A))``

    Output:

    ``[[-nan -nan], [-nan -nan]]``

    While ``print(sqrt_hmatrix(A))`` outputs

    ``[[1.58312395e-09, 3.15831240e-08],
    [3.15831240e-08, 1.58312395e-09]]``

    .. [7] N. J. Higham, "Computing real square
           roots of a real matrix", Linear
           Algebra Appl., 1987.
    """
    s, _, v = svd(matrix)
    return matmul(
        matmul(v, diag(tfsqrt(abs(s)))), v, adjoint_b=True,
        name="square_root",
    )
