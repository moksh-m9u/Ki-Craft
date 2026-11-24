"""Postgres package: the tables, the connection, and whole-document reads and writes.

``schema.py`` describes the tables, ``connection.py`` opens the database, and
``store.py`` reads and writes one document's rows. The migrations in
``migrations/`` create the tables. Import from the modules directly; this
file only marks the folder as a package.
"""
