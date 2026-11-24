"""Offline package: PDF to chunks, summaries and Postgres rows.

``extract.py`` partitions the PDF, ``chunk.py`` groups elements by title,
``enrich.py`` writes a summary per chunk, ``export.py`` saves the JSON export
and ``pipeline.py`` runs the steps and writes the store. Import from those
modules directly; this file only marks the folder as a package.
"""
