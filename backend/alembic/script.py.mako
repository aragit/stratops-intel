#%%
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "${up_revision}"
down_revision = "${down_revision}"
branch_labels = ${branch_labels}
depends_on = ${depends_on}


def upgrade() -> None:
    ${upgrades}


def downgrade() -> None:
    ${downgrades}