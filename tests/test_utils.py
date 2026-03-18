"""Unit tests for utility helpers."""

import pytest

from selfplay.env.agents import (
    BLineAgent,
    BlueSleepAgent,
    MeanderAgent,
    ReactRestoreAgent,
    RestoreDecoysAgent,
)
from selfplay.utils import make_scripted_agent


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("red_bline", BLineAgent),
        ("red_meander", MeanderAgent),
        ("blue_sleep", BlueSleepAgent),
        ("blue_react_restore", ReactRestoreAgent),
        ("blue_restore_decoys", RestoreDecoysAgent),
    ],
)
def test_make_scripted_agent_accepts_canonical_names(name, expected_type):
    agent = make_scripted_agent(name)
    assert isinstance(agent, expected_type)


@pytest.mark.parametrize("name", ["bline", "meander", "react_restore", "sleep"])
def test_make_scripted_agent_rejects_legacy_aliases(name):
    with pytest.raises(ValueError, match="Unknown scripted agent"):
        make_scripted_agent(name)
