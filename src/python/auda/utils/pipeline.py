import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from pkgutil import iter_modules
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableSequence,
    Self,
    Sequence,
    Tuple,
    Type,
    cast,
    get_origin,
)

# Type aliases for input/output value mappings
IOValueMap = Dict[str | Enum, Any]

# Type alias for input/output value objects with string keys
IOValueObject = Dict[str, Any]


def _to_plain_name(name: str | Enum) -> str:
    """Converts a name (str or Enum) to its plain string representation.

    Args:
        name: The name to convert, either as a string or an Enum member.

    Returns:
        The string representation of the name.
    """

    return cast(str, name.value if isinstance(name, Enum) else name)


@dataclass()
class IOSpec:
    """Specification for a single input or output port.

    Attributes:
        name: The name of the port.
        description: A human-readable description of the port.
        dtype: The expected type for the value on this port.
        required: Whether the port must be populated. For inputs, if True, an
            error will be raised if no value is provided; for outputs, if True,
            an error will be raised if no value is produced.
        default: The default value for this port if no value is provided (only
            applicable for input ports). If None, no default value is set.
    """

    name: str | Enum
    description: str = ''
    dtype: Any = Any
    required: bool = True
    default: Any = None

    def update(
        self,
        description: str | None = None,
        dtype: Any = None,
        required: bool | None = None,
        default: Any = None,
    ) -> 'IOSpec':
        """Returns a new IOSpec with updated attributes.

        Note: the attribute `name` cannot be updated.

        Args:
            description: New description for this port.
            dtype: New expected type for the value on this port.
            required: New required status for this port.
            default: New default value for this port.

        Returns:
            A new IOSpec instance with the updated attributes.
        """

        return IOSpec(
            name=self.name,
            description=description or self.description,
            dtype=dtype or self.dtype,
            required=self.required if required is None else required,
            default=self.default if default is None else default,
        )

    def optional(self, default: Any = None) -> 'IOSpec':
        """Retrieves the default value for this port.

        Returns:
            The default value for this port.
        """

        return self.update(default=default, required=False)

    def desc(self, description: str) -> 'IOSpec':
        """Returns a new IOSpec with an updated description.

        Args:
            description: New description for this port.

        Returns:
            A new IOSpec instance with the updated description.
        """

        return self.update(description=description)


class Step(ABC):
    """Abstract unit of work in a data-processing pipeline."""

    def __init__(self, spec: 'StepSpec', pipeline: 'Pipeline') -> None:
        """Initializes a step instance bound to a StepSpec instance.

        Args:
            spec: The declarative specification describing this step.
            pipeline: The Pipeline instance this step is part of.
        """

        self.spec = spec
        self.pipeline: Pipeline = pipeline
        self._inputs: IOValueObject = {}
        self._outputs: IOValueObject = {}

    def set_input(self, name: str | Enum, value: Any) -> None:
        """Assigns a single input value.

        Args:
            name: Input port name.
            value: Value to assign to the input port.

        Raises:
            KeyError: If the input port is not defined in the spec.
        """

        name = _to_plain_name(name)
        self._inputs[name] = value

    def get_input(self, name: str | Enum, check_port: bool = True) -> Any:
        """Retrieves an input value.

        Args:
            name: Input port name.
            check_port: Whether to validate the input against the spec.

        Returns:
            The value associated with the port, or the default value set in the
            port if the port is optional.

        Raises:
            KeyError: If the input port is not defined in the spec.
            ValueError: If the input is required but not provided.
        """

        name = _to_plain_name(name)

        if not check_port:
            value: Any = self._inputs.get(name)
            if value is None:
                raise KeyError(f"Input port '{name}' is not set.")

            return value

        port: IOSpec | None = self.spec.input_spec_map.get(name)
        if port is None:
            raise KeyError(f"Input port '{name}' is not defined in step spec.")

        value: Any = self._inputs.get(name)
        if value is None:
            if port.required:
                raise ValueError(
                    f"Input '{name}' is required but not provided."
                )

            return port.default

        return value

    def set_output(self, name: str | Enum, value: Any) -> None:
        """Assigns an output value.

        Args:
            name: Output port name.
            value: Value to assign to the output port.

        Raises:
            KeyError: If the output port is not defined in the spec.
            TypeError: If the value does not match the declared dtype.
        """

        name = _to_plain_name(name)
        port = self.spec.output_spec_map.get(name)
        if port is None:
            raise KeyError(f"Output '{name}' is not defined in step spec.")

        self._outputs[name] = value

    def get_output(self, name: str | Enum, check_port=False) -> Any:
        """Retrieves an output value.

        Args:
            name: Output port name.
            check_port: Whether to validate the output against the spec.

        Returns:
            The output value, or the default value set in the port if the port
            is optional.

        Raises:
            KeyError: If the output port is not defined in the spec.
            ValueError: If the output is required but has not been set.
        """

        name = _to_plain_name(name)

        if not check_port:
            value: Any = self._inputs.get(name)
            if value is None:
                raise KeyError(f"Input port '{name}' is not set.")

            return value

        port = self.spec.output_spec_map.get(name)
        if port is None:
            raise KeyError(f"Output '{name}' is not defined in step spec.")

        value = self._outputs.get(name)
        if value is None:
            if port.required:
                raise ValueError(f"Output '{name}' is required but not set.")
            return None

        return value

    def set_inputs(self, inputs: IOValueObject) -> None:
        """Assigns multiple input values at once.

        Args:
            inputs: Mapping of input port names to values.

        Raises:
            KeyError: If any input port is not defined in the spec.
        """

        for name, value in inputs.items():
            self.set_input(name, value)

    def get_outputs(self) -> IOValueObject:
        """
        Retrieves all output values as a dictionary.

        This method will return a copy of the internal outputs dictionary.

        Returns:
            A mapping of output port names to their assigned values.
        """

        return self._outputs.copy()

    @abstractmethod
    def run(self, **inputs) -> IOValueMap | None:
        """Executes the step’s work.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    def single_dispatch(
        self, to: str | Type, inputs: IOValueMap
    ) -> IOValueMap | None:
        """Alias for `run()` to support dispatching.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        return create_pipeline([to]).run([inputs])


@dataclass(frozen=True)
class StepSpec:
    """
    Declarative specification for constructing and validating a Step.

    Describes the step’s identity, kind, human-readable description, concrete
    implementation class, and I/O port specifications.

    Attributes:
        id: Unique identifier for the step within a pipeline or registry.
        description: Human-readable summary of what the step does.
        implementation: Concrete `Step` subclass implementing `run()`.
        input_specs: Mapping of input port names to their `IOSpec`.
        output_specs: Mapping of output port names to their `IOSpec`.
    """

    id: str
    description: str
    implementation: Type[Step]
    input_spec_map: Dict[str, IOSpec] = field(default_factory=dict)
    output_spec_map: Dict[str, IOSpec] = field(default_factory=dict)

    def instantiate(self, pipeline: 'Pipeline') -> Step:
        """Initializes a Step instance bound to this step spec.

        Args:
            pipeline: The Pipeline instance the step will be part of.

        Returns:
            A new instance of implementation defined in this step spec.
        """

        return self.implementation(self, pipeline)


class Pipeline:
    def __init__(self, step_specs: List[StepSpec]):
        """Initializes a Pipeline instance.

        Attributes:
            step_specs: A list of StepSpec instances representing the steps in
                the pipeline.
            callbacks: A list of callable functions to be executed after the
                pipeline run.
            context: A dictionary for storing shared data across steps.
        """

        self._step_specs: List[StepSpec] = step_specs
        self._callbacks: List[Callable[['Pipeline'], None]] = []
        self.context: IOValueObject = {}

    def get_value(self, name: str | Enum) -> Any:
        """Retrieves a value from the pipeline context.

        Args:
            name: The name of the value to retrieve.
        """

        return self.context.get(_to_plain_name(name))

    def run(
        self,
        step_inputs: List[IOValueMap] | None = None,
    ) -> None:
        """Runs the pipeline by executing each step in sequence.

        Args:
            step_inputs: Optional list of dictionaries containing input values
                for each step. Each dictionary corresponds to a step in the
                pipeline. If provided, these inputs will override the values in
                the pipeline context for the respective step.

        Raises:
            ValueError: If required inputs for any step are missing.
        """

        if step_inputs is None:
            step_inputs = []

        steps: List[Step] = [
            spec.instantiate(self) for spec in self._step_specs
        ]

        for i, step in enumerate(steps):
            # Update the context with step-specific inputs if provided
            if len(step_inputs) > i and step_inputs[i]:
                self.context.update(
                    {
                        _to_plain_name(name).upper(): value
                        for name, value in step_inputs[i].items()
                    }
                )

            try:
                self.run_step(step)
            except Exception as e:
                raise RuntimeError(
                    f"Error occurred while executing step '{step.spec.id}': {e}"
                ) from e

    def run_step(self, step: Step) -> None:
        # Traverse input specs to ensure required inputs are present
        # Raise an error if a required input is missing; set default values
        # for optional inputs that are not provided
        for input_name, input_spec in step.spec.input_spec_map.items():
            if input_name not in self.context:
                if input_spec.required:
                    raise ValueError(
                        f"Input '{input_name}' is required for step "
                        f"'{step.spec.id}' but not provided."
                    )
                else:
                    self.context[input_name] = input_spec.default

        step.set_inputs(self.context)

        # Prepare inputs arguments
        input_args: IOValueObject = {}
        parameters = inspect.signature(step.run).parameters
        for name, param in parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                # input_args[name] = step.get_input(name.upper())

                input_args[name] = self.convert_by_type(
                    param.annotation, step.get_input(name.upper())
                )

        # Execute the step and collect outputs
        outputs = step.run(**input_args)

        # Set outputs
        if outputs is not None:
            for name, value in outputs.items():
                step.set_output(_to_plain_name(name).upper(), value)

        # Traverse output specs to ensure required outputs are present
        # Raise an error if a required output is missing; set default values
        # for optional outputs that are not produced
        outputs = step.get_outputs()
        for output_name, output_spec in step.spec.output_spec_map.items():
            if output_name not in outputs:
                if output_spec.required:
                    raise ValueError(
                        f"Output '{output_name}' is required for step "
                        f"'{step.spec.id}' but was not produced."
                    )
                else:
                    step.set_output(output_name, output_spec.default)

        # Update the pipeline context with the step's outputs
        self.context.update(step.get_outputs())

    def schedule(self, callback: Callable[['Pipeline'], None]) -> None:
        """Schedules a callback to be executed after the pipeline run.

        Args:
            callback: A callable function to be executed after the pipeline run.
        """

        self._callbacks.append(callback)

    def execute_callbacks(self) -> None:
        """Executes all scheduled callbacks."""

        for callback in self._callbacks:
            callback(self)

    def reset(self) -> Self:
        """Resets the pipeline context and clears scheduled callbacks."""

        self._step_specs = [
            get_step_spec_by_id(step_spec.id) for step_spec in self._step_specs
        ]
        self._callbacks.clear()
        self.context.clear()

        return self

    def convert_by_type(self, dtype: Type, value: Any) -> Any:
        """Converts a value to a specified type using single dispatch.

        Args:
            to: The target type to convert the value to.
            value: The value to be converted.

        Returns:
            The converted value.
        """
        try:
            if dtype is str:
                return str(value)
            if dtype is int:
                return int(value)
            elif dtype is float:
                return float(value)
            elif dtype is bool:
                return bool(value)
            elif get_origin(dtype) is list:
                str_list = str(value).split(VALUES_DELIMITER)
                item_type = dtype.__args__[0]
                if item_type is str:
                    return str_list
                elif item_type is int:
                    return [int(item) for item in str_list]
                elif item_type is float:
                    return [float(item) for item in str_list]
                elif item_type is bool:
                    return [bool(item) for item in str_list]
                else:
                    return str_list
            else:
                return value
        except Exception as e:
            raise ValueError(
                f"Failed to convert value '{value}' to type '{dtype}': {e}"
            ) from e


# Registry for StepSpecs; maps from step ID to StepSpec
_step_specs: Dict[str, StepSpec] = {}

# Registry for StepSpecs by their implementation class
_step_specs_by_class: Dict[Type[Step], StepSpec] = {}


def register(spec: StepSpec) -> None:
    """Registers a StepSpec.

    Args:
        spec: The StepSpec to register.

    Raises:
        ValueError: If a StepSpec with the same ID is already registered.
    """

    if spec.id in _step_specs:
        raise ValueError(f'Duplicate StepSpec ID: {spec.id}')

    _step_specs[spec.id] = spec
    _step_specs_by_class[spec.implementation] = spec


def get_all_step_specs() -> List[StepSpec]:
    """Retrieves all registered StepSpecs."""

    return list(_step_specs.values())


def step(
    *,
    id: str,
    description: str,
    input_specs: List[IOSpec] | None = None,
    output_specs: List[IOSpec] | None = None,
):
    """Decorates a Step subclass to register a step spec for it.

    Args:
        id: Unique identifier for the step within a pipeline or registry.
        description: Human-readable summary of what the step does.
        input_specs: Mapping of input port names to their IOSpec.
        output_specs: Mapping of output port names to their IOSpec.
    """

    input_spec_map = {
        _to_plain_name(spec.name): spec for spec in (input_specs or [])
    }

    output_spec_map = {
        _to_plain_name(spec.name): spec for spec in (output_specs or [])
    }

    def _wrap(cls):
        spec = StepSpec(
            id=id,
            description=description,
            implementation=cls,
            input_spec_map=input_spec_map,
            output_spec_map=output_spec_map,
        )
        register(spec)

        return cls

    return _wrap


def scan_package(path: MutableSequence[str], package_name: str) -> None:
    """Scans a package for modules and subpackages to import.

    This for loop will import all modules and subpackages in the specified
    package except those whose names start with an underscore (_).

    Args:
        path: List of paths to scan for modules and subpackages.
        package_name: The name of the package to scan.
    """

    for _, name, is_package in iter_modules(path, prefix=f'{package_name}.'):
        if is_package and not name.startswith('_'):
            package = import_module(name)

            for _, module_name, _ in iter_modules(package.__path__, name + '.'):
                if module_name.rsplit('.', 1)[-1].startswith('_'):
                    continue

                import_module(module_name)


def get_step_spec_by_id(step_id: str) -> StepSpec:
    """Retrieves a registered StepSpec by its ID.

    This function will attempt to dynamically import the module if the
    `_module_name_getter` is set and returns a module name for the given step
    ID.

    Args:
        step_id: The unique identifier of the step.

    Returns:
        The StepSpec associated with the given ID.
    """
    if _module_name_getter is not None:
        module_name = _module_name_getter(step_id)
        if module_name is not None:
            import_module(module_name)

    step_spec = _step_specs.get(step_id)

    if step_spec is None:
        raise ValueError(f'Step spec not found: {step_id}')

    return step_spec


def get_step_spec_by_class(step_class: Type[Step]) -> StepSpec:
    """Retrieves a registered StepSpec by its implementation class.

    Args:
        step_class: The implementation class of the step.

    Returns:
        The StepSpec associated with the given implementation class.
    """

    step_spec = _step_specs_by_class.get(step_class)

    if step_spec is None:
        raise ValueError(f'Step spec not found for class: {step_class}')

    return step_spec


def create_pipeline(step_id_list: Sequence[str | Type[Step]]) -> Pipeline:
    """Creates a Pipeline instance from a list of StepSpec IDs.

    Args:
        step_id_list: A list of StepSpec IDs to include in the pipeline.

    Returns:
        A Pipeline instance containing the specified steps.
    """

    return Pipeline(
        [
            get_step_spec_by_id(step_id)
            if isinstance(step_id, str)
            else get_step_spec_by_class(step_id)
            for step_id in step_id_list
        ]
    )


STEP_STRS_DELIMITER = ' '
STEP_STR_PARAM_DELIMITER = ':'
INPUTS_STR_DELIMITER = ';'
INPUTS_STR_KEY_VALUE_DELIMITER = '='
VALUES_DELIMITER = ','


def create_step_str_list(pipe_str: str) -> List[str]:
    """Creates a list of step strings from a pipeline string.

    Args:
        pipe_str: A string representing the pipeline, with step IDs
            separated by spaces.

    Returns:
        A list of step ID strings.
    """

    return [
        step_str.strip()
        for step_str in pipe_str.split(STEP_STRS_DELIMITER)
        if step_str
    ]


def process_step_str(step_str: str) -> Tuple[str, IOValueObject]:
    """Processes a step string to extract the step ID and input values.

    Args:
        step_str: A string representing a step, potentially with input
            parameters.

    Returns:
        A tuple containing the step ID and a dictionary of input values.
    """

    inputs: IOValueObject = {}

    if STEP_STR_PARAM_DELIMITER in step_str:
        step_id, inputs_str = step_str.split(STEP_STR_PARAM_DELIMITER, 1)
        inputs_strs = inputs_str.split(INPUTS_STR_DELIMITER)

        raw_inputs = {}
        for input in inputs_strs:
            sp = input.split(INPUTS_STR_KEY_VALUE_DELIMITER, 1)
            raw_inputs[sp[0].upper()] = sp[1]

        inputs.update(raw_inputs)
    else:
        step_id = step_str

    return step_id, inputs


def parse_step_str_list(
    step_strs: List[str],
) -> Tuple[Pipeline, List[IOValueObject]]:
    """Creates a Pipeline instance from a list of step strings.

    Step IDs are case-insensitive and will be normalized to uppercase.

    Args:
        step_strs: A list of step strings.

    Returns:
        A Pipeline instance containing the specified steps.
    """

    step_ids: List[str] = []
    step_inputs: List[IOValueObject] = []
    for step_str in step_strs:
        step_id, inputs = process_step_str(step_str)
        step_ids.append(step_id)
        step_inputs.append(inputs)

    # Normalize step IDs to uppercase
    step_ids = [step_id.upper() for step_id in step_ids]

    return create_pipeline(step_ids), step_inputs


# Maps from step ID prefixes to their kinds
_prefix_kind_map: Dict[str, str] = {}

# Default kind if no prefix matches
UNKNOWN_KIND = 'Unknown'


def add_kind(prefix: str, kind: str) -> None:
    """Associates a kind with a step ID prefix.

    Args:
        prefix: The prefix of the step ID.
        kind: The kind to associate with the prefix.
    """

    _prefix_kind_map[prefix] = kind


def get_kind_by_id(step_id: str) -> str:
    """Retrieves the kind associated with a step ID based on its prefix.

    Args:
        step_id: The unique identifier of the step.

    Returns:
        The kind associated with the step ID, or an empty string if no kind
        is found.
    """

    for prefix, kind in _prefix_kind_map.items():
        if step_id.startswith(prefix):
            return kind

    return UNKNOWN_KIND


def get_kind(step_spec: StepSpec) -> str:
    """Retrieves the kind associated with a StepSpec based on its ID prefix.

    Args:
        step_spec: The StepSpec instance.

    Returns:
        The kind associated with the StepSpec, or an empty string if no kind
        is found.
    """

    return get_kind_by_id(step_spec.id)


# Function to retrieve module names for step IDs
_module_name_getter: Callable[[str], str | None] | None = None


def set_module_name_getter(getter: Callable[[str], str | None]) -> None:
    """Sets the function to retrieve module names for step IDs.

    Args:
        getter: A callable that takes a step ID and returns the corresponding
            module name, or None if not found.
    """

    global _module_name_getter
    _module_name_getter = getter
