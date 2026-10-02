
from tools import (
    forecast_tool,
    stockout_tool,
    inventory_tool,
)


def forecast_agent(question, context=None):
    context = context or {}
    return {
        "agent": "Forecast Agent",
        "result": forecast_tool(question, context),
    }


def stockout_agent(question, context=None):
    context = context or {}
    return {
        "agent": "Stockout Agent",
        "result": stockout_tool(question, context),
    }


def inventory_agent(question, context=None):
    context = context or {}
    return {
        "agent": "Inventory Agent",
        "result": inventory_tool(question, context),
    }


def policy_agent(question):
    from rag import retrieve_policy

    return {
        "agent": "Policy Agent",
        "result": retrieve_policy(question, k=3),
    }
