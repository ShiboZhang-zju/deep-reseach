"""Initial schema — all pre-Phase-0 tables.

Revision ID: 0000_initial_schema
Revises: 
Create Date: 2026-07-23

This migration creates ALL tables that existed before the Phase 0-2 refactoring.
A fresh database should start here and chain through all subsequent migrations.
"""

from alembic import op
import sqlalchemy as sa


revision = '0000_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # research_tasks
    op.create_table(
        'research_tasks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('user_input', sa.Text(), nullable=False),
        sa.Column('normalized_topic', sa.Text()),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('current_round', sa.Integer(), server_default='0'),
        sa.Column('max_rounds', sa.Integer(), server_default='5'),
        sa.Column('stop_reason', sa.Text()),
        sa.Column('state_json', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )

    # papers
    op.create_table(
        'papers',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('abstract', sa.Text()),
        sa.Column('authors_json', sa.Text()),
        sa.Column('year', sa.Integer()),
        sa.Column('venue', sa.Text()),
        sa.Column('doi', sa.Text()),
        sa.Column('arxiv_id', sa.Text()),
        sa.Column('semantic_scholar_id', sa.Text()),
        sa.Column('openalex_id', sa.Text()),
        sa.Column('url', sa.Text()),
        sa.Column('pdf_url', sa.Text()),
        sa.Column('citation_count', sa.Integer(), server_default='0'),
        sa.Column('sources_json', sa.Text()),
        sa.Column('raw_json', sa.Text()),
        sa.Column('normalized_title', sa.Text()),
        sa.Column('title_hash', sa.String()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_papers_doi', 'papers', ['doi'], unique=True, sqlite_where=sa.text('doi IS NOT NULL'))
    op.create_index('idx_papers_arxiv', 'papers', ['arxiv_id'], unique=True, sqlite_where=sa.text('arxiv_id IS NOT NULL'))
    op.create_index('idx_papers_s2', 'papers', ['semantic_scholar_id'], unique=True, sqlite_where=sa.text('semantic_scholar_id IS NOT NULL'))
    op.create_index('idx_papers_openalex', 'papers', ['openalex_id'], unique=True, sqlite_where=sa.text('openalex_id IS NOT NULL'))
    op.create_index('idx_papers_year', 'papers', ['year'])
    op.create_index('idx_papers_citation', 'papers', ['citation_count'])
    op.create_index('idx_papers_title_hash_idx', 'papers', ['title_hash'])

    # task_papers
    op.create_table(
        'task_papers',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('discovered_round', sa.Integer(), nullable=False),
        sa.Column('relevance_score', sa.Float()),
        sa.Column('authority_score', sa.Float()),
        sa.Column('recency_score', sa.Float()),
        sa.Column('novelty_score', sa.Float()),
        sa.Column('idea_potential_score', sa.Float()),
        sa.Column('final_score', sa.Float()),
        sa.Column('priority', sa.String()),
        sa.Column('reason', sa.Text()),
        sa.Column('summary', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_task_papers_unique', 'task_papers', ['task_id', 'paper_id'], unique=True)

    # research_rounds
    op.create_table(
        'research_rounds',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('round_number', sa.Integer(), nullable=False),
        sa.Column('queries_json', sa.Text()),
        sa.Column('papers_found', sa.Integer(), server_default='0'),
        sa.Column('new_papers', sa.Integer(), server_default='0'),
        sa.Column('duplicate_rate', sa.Float()),
        sa.Column('summary', sa.Text()),
        sa.Column('knowledge_gaps_json', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_rounds_unique', 'research_rounds', ['task_id', 'round_number'], unique=True)

    # reports
    op.create_table(
        'reports',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('content_markdown', sa.Text()),
        sa.Column('content_json', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )

    # research_ideas
    op.create_table(
        'research_ideas',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('title', sa.Text()),
        sa.Column('description', sa.Text()),
        sa.Column('motivation', sa.Text()),
        sa.Column('method_sketch', sa.Text()),
        sa.Column('expected_contribution', sa.Text()),
        sa.Column('novelty', sa.Float()),
        sa.Column('feasibility', sa.Float()),
        sa.Column('significance', sa.Float()),
        sa.Column('evidence_support', sa.Float()),
        sa.Column('differentiation', sa.Float()),
        sa.Column('experimentability', sa.Float()),
        sa.Column('potential_impact', sa.Float()),
        sa.Column('risk', sa.Float()),
        sa.Column('final_score', sa.Float()),
        sa.Column('decision', sa.String()),
        sa.Column('related_paper_ids_json', sa.Text()),
        sa.Column('user_selected', sa.Boolean(), server_default='0'),
        sa.Column('idea_status', sa.String(), server_default='active'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_ideas_task_status', 'research_ideas', ['task_id', 'idea_status'])

    # experiment_plans
    op.create_table(
        'experiment_plans',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('idea_id', sa.String(), sa.ForeignKey('research_ideas.id'), nullable=False),
        sa.Column('hypothesis', sa.Text()),
        sa.Column('dataset', sa.Text()),
        sa.Column('baselines', sa.Text()),
        sa.Column('metrics', sa.Text()),
        sa.Column('steps_markdown', sa.Text()),
        sa.Column('steps_json', sa.Text()),
        sa.Column('risks', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )

    # agent_traces
    op.create_table(
        'agent_traces',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('step_name', sa.Text(), nullable=False),
        sa.Column('step_type', sa.String(), nullable=False),
        sa.Column('round_number', sa.Integer()),
        sa.Column('input_json', sa.Text()),
        sa.Column('output_json', sa.Text()),
        sa.Column('llm_tokens_used', sa.Integer()),
        sa.Column('duration_ms', sa.Integer()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )

    # user_feedbacks
    op.create_table(
        'user_feedbacks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('feedback_type', sa.String(), nullable=False),
        sa.Column('content', sa.Text()),
        sa.Column('selected_idea_ids_json', sa.Text()),
        sa.Column('need_more_research', sa.Boolean(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )

    # paper_chunks
    op.create_table(
        'paper_chunks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('section', sa.Text(), server_default='unknown'),
        sa.Column('chunk_type', sa.Text(), server_default='text'),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('image_paths_json', sa.Text(), server_default='[]'),
        sa.Column('page_number', sa.Integer(), server_default='0'),
        sa.Column('word_count', sa.Integer(), server_default='0'),
        sa.Column('has_pdf', sa.Boolean(), server_default='0'),
        sa.Column('extraction_method', sa.Text(), server_default='pymupdf_inline'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_paper_chunks_paper', 'paper_chunks', ['paper_id'])
    op.create_index('idx_paper_chunks_section', 'paper_chunks', ['section'])
    op.create_index('idx_paper_chunks_type', 'paper_chunks', ['chunk_type'])

    # paper_analyses
    op.create_table(
        'paper_analyses',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('problem', sa.Text(), server_default=''),
        sa.Column('method_detail', sa.Text(), server_default=''),
        sa.Column('experiment_setup', sa.Text(), server_default=''),
        sa.Column('key_results', sa.Text(), server_default=''),
        sa.Column('limitations', sa.Text(), server_default=''),
        sa.Column('extendable_components', sa.Text(), server_default=''),
        sa.Column('source_sections', sa.Text(), server_default='{}'),
        sa.Column('has_full_text', sa.Boolean(), server_default='0'),
        sa.Column('analysis_tokens', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_paper_analyses_task', 'paper_analyses', ['task_id'])
    op.create_index('idx_paper_analyses_paper', 'paper_analyses', ['paper_id'])

    # paper_citations
    op.create_table(
        'paper_citations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('source_paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('target_paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('relation_type', sa.String(), nullable=False),
        sa.Column('weight', sa.Float(), server_default='1.0'),
        sa.Column('source_task_id', sa.String(), sa.ForeignKey('research_tasks.id')),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_citations_source', 'paper_citations', ['source_paper_id'])
    op.create_index('idx_citations_target', 'paper_citations', ['target_paper_id'])
    op.create_index('idx_citations_type', 'paper_citations', ['relation_type'])
    op.create_index('idx_citations_unique', 'paper_citations', ['source_paper_id', 'target_paper_id', 'relation_type'], unique=True)

    # wiki_pages
    op.create_table(
        'wiki_pages',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('page_type', sa.String(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content_markdown', sa.Text(), server_default=''),
        sa.Column('paper_ids_json', sa.Text(), server_default='[]'),
        sa.Column('links_json', sa.Text(), server_default='[]'),
        sa.Column('contradictions_json', sa.Text(), server_default='[]'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_wiki_task_type', 'wiki_pages', ['task_id', 'page_type'])
    op.create_index('idx_wiki_task_title', 'wiki_pages', ['task_id', 'title'])


def downgrade() -> None:
    op.drop_table('wiki_pages')
    op.drop_table('paper_citations')
    op.drop_table('paper_analyses')
    op.drop_table('paper_chunks')
    op.drop_table('user_feedbacks')
    op.drop_table('agent_traces')
    op.drop_table('experiment_plans')
    op.drop_table('research_ideas')
    op.drop_table('reports')
    op.drop_table('research_rounds')
    op.drop_table('task_papers')
    op.drop_table('papers')
    op.drop_table('research_tasks')
