# Retail Demand: Agent + RAG starter (prototype)

This is a **working starter/prototype**, not the final teammate-integrated implementation. The policy files are sample text and must be replaced by the team's approved policies. The retrieval uses TF-IDF so it can run simply in Colab; if your rubric requires dense embeddings + FAISS/Chroma, replace `rag.py`'s retrieval backend while keeping `retrieve_policy(query, k=3)` as the interface.

## Run in Google Colab
1. Upload/extract this folder or place these files in one Colab directory.
2. Install: `!pip install -r requirements.txt`
3. Run tests: `!python -m unittest -v test_agent.py`
4. Demo: `!python agent.py`

## When teammates finish
- Replace the six sample `.md` files with the approved policy files (keep `.md` format, or update the loader).
- Copy the teammates' `forecasting.py`, `stockout.py`, `reorder.py`, and their required CSV/model files into the same working directory (or update paths/imports).
- In `tools.py`, set `SAMPLE_MODE = False` and verify the real function inputs. The forecast adapter assumes context keys `store_id`, `product_id`, `days`; stockout expects `store_id`, `product_id`, `forecast_demand`, `current_stock`; reorder expects `forecast_demand`, `current_stock`, `lead_time`, `safety_stock`.
- Replace `promo_tool` with the actual teammate function once available.
- Note: the supplied forecasting module trains/loads data at import and uses relative paths. Run from the directory containing the required CSV, and test its import separately.

## Sample context
`answer(question, context={...})` accepts context values; this simple prototype does not extract IDs from natural language. A UI or LLM layer can populate context later. Never put API keys in source code or commit them to Git.
