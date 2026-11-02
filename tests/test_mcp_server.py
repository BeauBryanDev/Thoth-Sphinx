import json
import os
import sys
from pathlib import Path

import anyio
import psycopg
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from evidence_store import retrieve

ROOT = Path(__file__).resolve().parent.parent
TOOLS = {'search_parallels', 'lookup_lemma', 'get_context', 'list_periods', 'lookup_entity'}


@pytest.fixture(scope='module', autouse=True)
def database():
    try:
        retrieve.query('SELECT 1 FROM entities LIMIT 1')
    except psycopg.Error:
        pytest.skip('evidence database is not running or has no graph')


async def session_run(work):
    parameters = StdioServerParameters(command=sys.executable,
                                       args=['-m', 'evidence_store.mcp_server'], cwd=str(ROOT),
                                       env=dict(os.environ))
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await work(session)


def payload(result):
    return json.loads(next(block.text for block in result.content if block.type == 'text'))


def test_a_real_client_lists_every_tool_as_read_only():
    async def work(session):
        return (await session.list_tools()).tools

    listed = anyio.run(session_run, work)
    assert {tool.name for tool in listed} == TOOLS
    assert all(tool.description and tool.annotations.read_only_hint for tool in listed)


def test_a_real_client_follows_a_reading_through_every_tool():
    async def work(session):
        found = payload(await session.call_tool('search_parallels',
                                                 {'translit': 'ḥtp-ḏi̯ nswt wsjr nb ḏdw'}))
        context = payload(await session.call_tool('get_context',
                                                  {'sentence_id': found['hits'][0]['sentence_id']}))
        lemma = payload(await session.call_tool('lookup_lemma', {'text': 'nṯr'}))
        periods = payload(await session.call_tool('list_periods', {}))
        entity = payload(await session.call_tool('lookup_entity', {'name': 'Osiris'}))
        return found, context, lemma, periods, entity

    found, context, lemma, periods, entity = anyio.run(session_run, work)
    assert found['verdict'] == 'strong'
    assert context['sentences']
    assert lemma['found']
    assert periods['periods']
    assert entity['entities'][0]['name'] == 'Osiris'


def test_a_bad_request_returns_an_error_the_agent_can_read():
    async def work(session):
        return await session.call_tool('search_parallels', {'translit': 'nswt', 'date': 'Riech'})

    result = anyio.run(session_run, work)
    assert result.is_error is True
    assert 'unknown date period' in result.content[0].text
