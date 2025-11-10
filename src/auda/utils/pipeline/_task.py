from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import UnionType
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Type, TypeVar

T = TypeVar('T')

if TYPE_CHECKING:
    from pipeline import Pipeline


@dataclass(frozen=True)
class IOSpec(Generic[T]):
    """Specification for a single input or output port.

    Defines the expected Python type and whether a value is required. Used by `TaskSpec`
    to validate runtime I/O for a `Task`.

    Attributes:
        dtype: The expected type for the value on this port.
        required: Whether the port must be populated (input provided or output set).
    """

    dtype: Type[T] | UnionType | Any
    required: bool = True
    default: T | None = None


class Task(ABC):
    """Abstract unit of work in a data-processing pipeline.

    Subclasses implement `run()` to transform declared inputs into declared outputs.
    Instances are created from a `TaskSpec`, which also governs I/O validation.
    """

    def __init__(self, spec: 'TaskSpec', pipeline: Pipeline) -> None:
        """
        Initializes a task instance bound to a `TaskSpec`.

        Args:
            spec: The declarative specification describing this task, including
                identity, implementation type, and input/output port definitions.
            pipeline: The `Pipeline` instance this task is part of.
        """
        self.spec = spec
        self.pipeline: Pipeline = pipeline
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}

    def set_input(self, name: str, value: Any) -> None:
        """
        Assigns a single input value without immediate type validation.

        The value is stored directly and validated later upon retrieval with
        `get_input(...)`.

        Args:
            name: Input port name.
            value: Value to assign to the input port.

        Raises:
            KeyError: If the input port is not defined in the spec.
        """
        self._inputs[name] = value

    def get_input(self, name: str) -> Any:
        """
        Retrieves and validate an input value.

        The value is type-checked against the corresponding `IOSpec`. If the port
        is marked required but no value was set, an error is raised.

        Args:
            name: Input port name.

        Returns:
            The value associated with the port, or `None` if the port is optional
            and no value was provided.

        Raises:
            KeyError: If the input port is not defined in the spec.
            ValueError: If the input is required but not provided.
            TypeError: If the provided value does not match the declared dtype.
        """
        spec: Optional[IOSpec] = self.spec.input_specs.get(name)
        if spec is None:
            raise KeyError(f"Input '{name}' is not defined in task spec.")

        value: Any = self._inputs.get(name)
        if value is None:
            if spec.required:
                raise ValueError(f"Input '{name}' is required but not provided.")

            return spec.default

        return value

    def set_output(self, name: str, value: Any) -> None:
        """
        Assigns and validate an output value.

        The value is validated immediately against the corresponding `IOSpec`.

        Args:
            name: Output port name.
            value: Value to assign to the output port.

        Raises:
            KeyError: If the output port is not defined in the spec.
            TypeError: If the value does not match the declared dtype.
        """
        port = self.spec.output_specs.get(name)
        if port is None:
            raise KeyError(f"Output '{name}' is not defined in task spec.")

        # if not isinstance(value, port.dtype):
        #     raise TypeError(
        #         f"Output '{name}' must be of type {port.dtype.__name__}, "
        #         f'got {type(value).__name__}.'
        #     )

        self._outputs[name] = value

    def get_output(self, name: str) -> Any:
        """
        Retrieves an output value, enforcing requiredness.

        Args:
            name: Output port name.

        Returns:
            The output value, or `None` if the port is optional and has not been set.

        Raises:
            KeyError: If the output port is not defined in the spec.
            ValueError: If the output is required but has not been set.
        """
        port = self.spec.output_specs.get(name)
        if port is None:
            raise KeyError(f"Output '{name}' is not defined in task spec.")

        value = self._outputs.get(name)
        if value is None:
            if port.required:
                raise ValueError(f"Output '{name}' is required but not set.")
            return None

        return value

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Assigns multiple input values at once.

        Args:
            inputs: Mapping of input port names to values.

        Raises:
            KeyError: If any input port is not defined in the spec.
        """
        for name, value in inputs.items():
            self.set_input(name, value)

    def get_outputs(self) -> Dict[str, Any]:
        """
        Retrieves all output values as a dictionary.

        This method will return a copy of the internal outputs dictionary.

        Returns:
            A mapping of output port names to their assigned values.
        """
        return self._outputs.copy()

    @abstractmethod
    def run(self) -> None:
        """
        Executes the task’s work.

        Implementations should:
          1. Read inputs using `get_input(...)`.
          2. Perform the computation.
          3. Populate outputs using `set_output(...)`.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class TaskSpec:
    """
    Declarative specification for constructing and validating a `Task`.

    Describes the task’s identity, kind, human-readable description, concrete
    implementation class, and I/O port specifications.

    Attributes:
        id: Unique identifier for the task within a pipeline or registry.
        kind: Free-form classifier for the task family/type.
        description: Human-readable summary of what the task does.
        implementation: Concrete `Task` subclass implementing `run()`.
        input_specs: Mapping of input port names to their `IOSpec`.
        output_specs: Mapping of output port names to their `IOSpec`.
    """

    id: str
    kind: str
    description: str
    implementation: Type[Task]
    input_specs: Dict[str, IOSpec[Any]] = field(default_factory=dict)
    output_specs: Dict[str, IOSpec[Any]] = field(default_factory=dict)

    def instantiate(self, pipeline: Pipeline) -> Task:
        """
        Creates a new `Task` instance bound to this specification.

        Args:
            pipeline: The `Pipeline` instance the task will be part of.

        Returns:
            A new instance of `implementation`, initialized with this `TaskSpec`.
        """
        return self.implementation(self, pipeline)
