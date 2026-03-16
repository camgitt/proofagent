"""Tests for multi-turn conversation testing."""

import pytest

from proofagent import Conversation, ConversationTurn, LLMResult, expect


# --- Conversation construction ---

def test_conversation_from_tuples():
    conv = Conversation(turns=[
        ("What's 2+2?", LLMResult(text="4")),
        ("And times 3?", LLMResult(text="12")),
    ])
    assert len(conv.turns) == 2
    assert isinstance(conv.turns[0], ConversationTurn)
    assert conv.turns[0].user_message == "What's 2+2?"
    assert conv.turns[0].result.text == "4"


# --- turn() access ---

def test_turn_access_and_negative_index():
    conv = Conversation(turns=[
        ("First", LLMResult(text="A")),
        ("Second", LLMResult(text="B")),
        ("Third", LLMResult(text="C")),
    ])
    assert conv.turn(0).result.text == "A"
    assert conv.turn(-1).result.text == "C"
    expect(conv.turn(1).result).contains("B")


# --- total_cost and total_latency ---

def test_total_cost_and_latency():
    conv = Conversation(turns=[
        ("Q1", LLMResult(text="A1", cost=0.01, latency=0.5)),
        ("Q2", LLMResult(text="A2", cost=0.02, latency=1.0)),
        ("Q3", LLMResult(text="A3", cost=0.03, latency=0.8)),
    ])
    assert conv.total_cost == pytest.approx(0.06)
    assert conv.total_latency == pytest.approx(2.3)


# --- turn_count assertion ---

def test_turn_count_pass():
    conv = Conversation(turns=[
        ("Q1", LLMResult(text="A1")),
        ("Q2", LLMResult(text="A2")),
    ])
    expect(conv).turn_count(2)


def test_turn_count_fail():
    conv = Conversation(turns=[
        ("Q1", LLMResult(text="A1")),
    ])
    with pytest.raises(AssertionError, match="Expected 3 turns but conversation has 1"):
        expect(conv).turn_count(3)


# --- all_turns_cost_under assertion ---

def test_all_turns_cost_under_pass():
    conv = Conversation(turns=[
        ("Q1", LLMResult(text="A1", cost=0.02)),
        ("Q2", LLMResult(text="A2", cost=0.03)),
    ])
    expect(conv).all_turns_cost_under(0.10)


def test_all_turns_cost_under_fail():
    conv = Conversation(turns=[
        ("Q1", LLMResult(text="A1", cost=0.50)),
        ("Q2", LLMResult(text="A2", cost=0.60)),
    ])
    with pytest.raises(AssertionError, match="Expected total conversation cost under"):
        expect(conv).all_turns_cost_under(1.00)


# --- no_turn_refused assertion ---

def test_no_turn_refused_pass():
    conv = Conversation(turns=[
        ("What's 2+2?", LLMResult(text="4")),
        ("And times 3?", LLMResult(text="12")),
    ])
    expect(conv).no_turn_refused()


def test_no_turn_refused_fail():
    conv = Conversation(turns=[
        ("What's 2+2?", LLMResult(text="4")),
        ("How to hack?", LLMResult(text="I can't help with that.")),
    ])
    with pytest.raises(AssertionError, match="Turn 1 appears to be a refusal"):
        expect(conv).no_turn_refused()


# --- chaining ---

def test_conversation_assertion_chaining():
    conv = Conversation(turns=[
        ("Q1", LLMResult(text="A1", cost=0.01)),
        ("Q2", LLMResult(text="A2", cost=0.02)),
        ("Q3", LLMResult(text="A3", cost=0.01)),
    ])
    (
        expect(conv)
        .turn_count(3)
        .all_turns_cost_under(0.10)
        .no_turn_refused()
    )
