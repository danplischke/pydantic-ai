from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, cast, overload

from pydantic import TypeAdapter
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai import (
    messages as _messages,
)
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.messages import UserContent
from pydantic_ai.run import AgentRunResult

from .._run_context import AgentDepsT, RunContext
from ..agent import AbstractAgent
from ..exceptions import UserError
from ..tools import (
    GenerateToolJsonSchema,
    ToolDefinition,
)
from .abstract import AbstractToolset, ToolsetTool

CallerDepsT = TypeVar('CallerDepsT')

HandoffDepsT = TypeVar('HandoffDepsT')
HandoffInputModelT = TypeVar('HandoffInputModelT')
HandoffOutputDataT = TypeVar('HandoffOutputDataT')

HandoffDepsFunc = Callable[[CallerDepsT, HandoffInputModelT], Awaitable[HandoffDepsT]]
"""A function that takes an input model and dependencies and returns the agent dependencies."""

HandoffUserPromptFunc = Callable[
    [CallerDepsT, HandoffInputModelT], Awaitable[str | Sequence[_messages.UserContent] | None]
]
"""A function that takes an input model and dependencies and returns the user prompt for the agent."""

HandoffRunFunc = Callable[
    [
        RunContext[CallerDepsT],
        AbstractAgent[HandoffDepsT, HandoffOutputDataT],
        HandoffDepsT,
        str | Sequence[UserContent] | None,
    ],
    Awaitable[AgentRunResult[HandoffOutputDataT] | HandoffOutputDataT],
]
"""A function that takes a run context, an agent, dependencies, and an input model and returns the result of the agent run."""


@dataclass
class HandoffToolsetTool(
    ToolsetTool[CallerDepsT], Generic[CallerDepsT, HandoffInputModelT, HandoffDepsT, HandoffOutputDataT]
):
    """A tool definition for an Agent Tool."""

    agent: AbstractAgent[HandoffDepsT, HandoffOutputDataT]

    input_type: type[HandoffInputModelT] | None = None

    deps_func: HandoffDepsFunc[CallerDepsT, HandoffInputModelT, HandoffDepsT] | None = None
    user_prompt_func: HandoffUserPromptFunc[CallerDepsT, HandoffInputModelT] | None = None

    run_func: HandoffRunFunc[CallerDepsT, HandoffDepsT, HandoffOutputDataT] | None = None

    async def _build_deps(self, ctx: RunContext[CallerDepsT], input: HandoffInputModelT) -> HandoffDepsT:
        if self.deps_func is None:
            return cast(HandoffDepsT, ctx.deps)

        return await self.deps_func(ctx.deps, input)

    async def _build_user_prompt(
        self, ctx: RunContext[CallerDepsT], input: HandoffInputModelT
    ) -> str | Sequence[_messages.UserContent] | None:
        if self.user_prompt_func is None:
            return format_as_xml(obj=input)

        return await self.user_prompt_func(ctx.deps, input)

    async def convert_args_to_input_type(self, args: dict[str, Any]) -> HandoffInputModelT:
        if self.input_type is None:
            raise ValueError('Input type is not set')

        type_adapter = TypeAdapter(self.input_type)
        return type_adapter.validate_python(args)

    async def call(self, ctx: RunContext[CallerDepsT], input: HandoffInputModelT) -> HandoffOutputDataT:
        handoff_deps: HandoffDepsT = await self._build_deps(ctx, input)

        handoff_user_prompt: str | Sequence[UserContent] | None = await self._build_user_prompt(ctx, input)

        if self.run_func is not None:
            func_result: AgentRunResult[HandoffOutputDataT] | HandoffOutputDataT = await self.run_func(
                ctx, self.agent, handoff_deps, handoff_user_prompt
            )
            if isinstance(func_result, AgentRunResult):
                return cast(HandoffOutputDataT, func_result.output)
            return func_result

        agent_run_result: AgentRunResult[HandoffOutputDataT] = await self.agent.run(
            user_prompt=handoff_user_prompt,
            deps=handoff_deps,
        )

        return agent_run_result.output

    def as_toolset_tool(self) -> ToolsetTool[CallerDepsT]:
        return ToolsetTool[CallerDepsT](
            toolset=self.toolset,
            tool_def=self.tool_def,
            max_retries=self.max_retries,
            args_validator=self.args_validator,
        )


@dataclass
class Handoff(Generic[CallerDepsT, HandoffInputModelT, HandoffDepsT, HandoffOutputDataT]):
    agent: AbstractAgent[CallerDepsT, HandoffOutputDataT]
    input_type: type[HandoffInputModelT] = str
    name: str | None = None
    description: str | None = None
    deps_func: HandoffDepsFunc[CallerDepsT, HandoffInputModelT, HandoffDepsT] | None = None
    user_prompt_func: HandoffUserPromptFunc[CallerDepsT, HandoffInputModelT] | None = None
    run_func: HandoffRunFunc[CallerDepsT, HandoffDepsT, HandoffOutputDataT] | None = None

    @classmethod
    def from_agent(
            cls,
            agent: AbstractAgent[CallerDepsT, HandoffOutputDataT],
    ) -> Handoff[CallerDepsT, HandoffInputModelT, CallerDepsT, HandoffOutputDataT]:
        """Create a Handoff from an Agent with the same dependencies as the caller."""
        if not agent.name:
            raise ValueError('Provide an Agent with a name')

        return Handoff(
            agent=agent,
            name=agent.name,
        )

    def as_handoff_toolset_tool(
        self,
        toolset: HandoffToolset[CallerDepsT],
    ) -> HandoffToolsetTool[CallerDepsT, HandoffInputModelT, HandoffDepsT, HandoffOutputDataT]:
        """Convert this Handoff into an HandoffToolsetTool by using the toolset's add_agent method."""
        if not (tool_name := self.name or self.agent.name):
            raise ValueError('Provide either `name` or an Agent with a name')

        agent_tool_def: ToolDefinition

        type_adapter = TypeAdapter(self.input_type)

        input_type_schema = type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
        input_type_schema['properties']['input'] = input_type_schema

        agent_tool_def = ToolDefinition(
            name=tool_name,
            description=self.description,
            parameters_json_schema=input_type_schema,
        )

        agent_toolset_tool: HandoffToolsetTool[AgentDepsT, Any, Any, Any] = HandoffToolsetTool(
            toolset=toolset,
            max_retries=toolset.max_retries,
            tool_def=agent_tool_def,
            args_validator=SchemaValidator(schema=core_schema.any_schema()),
            agent=self.agent,
            input_type=self.input_type,
            deps_func=self.deps_func,
            user_prompt_func=self.user_prompt_func,
            run_func=self.run_func,
        )

        return agent_toolset_tool


class HandoffToolset(AbstractToolset[AgentDepsT]):
    """A toolset that lets Agents be used as tools.

    See [toolset docs](../toolsets.md#agent-toolset) for more information.
    """

    max_retries: int

    agent_tools: dict[str, HandoffToolsetTool[AgentDepsT, Any, Any, Any]]

    _id: str | None

    def __init__(
        self,
        agent_tools: Sequence[HandoffToolsetTool[AgentDepsT, Any, Any, Any]] | None = None,
        max_retries: int = 1,
        *,
        id: str | None = None,
    ):
        """Build a new agent toolset.

        Args:
            agent_tools: The tools to add to the toolset.
            max_retries: The maximum number of retries for each tool during a run.
            id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the toolset's activities within the workflow.
        """
        self.max_retries = max_retries
        self._id = id

        self.agent_tools = {}
        for agent_tool in agent_tools or []:
            self.add_agent_tool(agent_tool)

    @property
    def id(self) -> str | None:
        return self._id

    def add_agent_tool(
        self,
        agent_tool: HandoffToolsetTool[AgentDepsT, Any, Any, Any],
    ) -> None:
        """Add an Agent as a tool to the toolset."""
        self.agent_tools[agent_tool.tool_def.name] = agent_tool

    def add_handoff(
        self,
        handoff: Handoff[AgentDepsT, HandoffInputModelT, HandoffDepsT, HandoffOutputDataT] | AbstractAgent[AgentDepsT, HandoffOutputDataT],
    ) -> None:
        """Add a Handoff as a tool to the toolset."""
        if isinstance(handoff, AbstractAgent):
            handoff = Handoff.from_agent(handoff)
        handoff_tool = handoff.as_handoff_toolset_tool(self)
        self.agent_tools[handoff_tool.tool_def.name] = handoff_tool

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {name: tool.as_toolset_tool() for name, tool in self.agent_tools.items()}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not (agent_tool := self.agent_tools.get(name)):
            raise UserError(f'Unknown tool: {name!r}')

        if agent_tool.input_type is None:
            return await agent_tool.call(ctx, input={})

        agent_input = await agent_tool.convert_args_to_input_type(tool_args)

        return await agent_tool.call(ctx, input=agent_input)
