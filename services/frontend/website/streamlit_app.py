import os
import uuid
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="PDF Ingest & Search", page_icon="🔎", layout="wide")

def make_request_id() -> str:
    return str(uuid.uuid4())

def backend_health() -> tuple[bool, str]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def post_search(query: str) -> tuple[int, dict]:
    headers = {"x-request-id": make_request_id()}
    r = requests.post(
        f"{BACKEND_URL}/search/",
        json={"query": query},
        headers=headers,
        timeout=60,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def post_ingest_dir(dir_path: str) -> tuple[int, dict]:
    headers = {"x-request-id": make_request_id()}
    # Backend accepts a text form field named "input"
    r = requests.post(
        f"{BACKEND_URL}/ingest/",
        data={"input": dir_path},
        headers=headers,
        timeout=300,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def post_ingest_files(uploaded_files) -> tuple[int, dict]:
    headers = {"x-request-id": make_request_id()}
    files = []
    for uf in uploaded_files:
        files.append(
            ("input", (uf.name, uf.getvalue(), "application/pdf"))
        )
    r = requests.post(
        f"{BACKEND_URL}/ingest/",
        files=files,
        headers=headers,
        timeout=300,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def normalize_backend_error(payload: dict) -> dict:
    # FastAPI often returns {"detail": {...}} for HTTPException
    if isinstance(payload, dict) and "detail" in payload and isinstance(payload["detail"], dict):
        return payload["detail"]
    return payload

def reset_state(keys: list[str]) -> None:
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("PDF Ingest & Semantic Search")
    st.caption(f"Backend: {BACKEND_URL}")

with col2:
    ok, msg = backend_health()
    st.write("")
    st.write("")
    st.status(f"Backend: {msg}", state="complete" if ok else "error")

st.divider()

# Toggle between Search & Ingest
search_mode = st.toggle("Search mode", value=True, help="Turn off to switch to Ingest mode")

if search_mode:
    st.subheader("Search")

    query = st.text_area("Query", placeholder="Type your question…", height=90, key="search_query")
    c1, c2 = st.columns([1, 5])
    with c1:
        do_search = st.button("Search", type="primary", use_container_width=True)

    if do_search:
        q = (query or "").strip()
        if not q:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching…"):
                status, payload = post_search(q)
            payload = normalize_backend_error(payload)
            st.session_state["search_last_response"] = {"status": status, "data": payload}

    resp = st.session_state.get("search_last_response")
    if resp:
        st.markdown("### Response")
        st.code(resp, language="json")

        status = resp["status"]
        data = resp["data"] if isinstance(resp["data"], dict) else {}

        if status == 200 and isinstance(data.get("results"), list):
            results = data["results"]
            st.markdown(f"### Results ({len(results)})")

            for idx, r in enumerate(results, start=1):
                doc = r.get("document", "")
                score = r.get("score", 0)
                content = r.get("content", "")

                title = f"{idx}. {doc} — score: {score:.4f}" if isinstance(score, (int, float)) else f"{idx}. {doc}"
                with st.expander(title, expanded=(idx == 1)):
                    top = st.columns([1, 1, 6])
                    top[0].metric("Rank", idx)
                    if isinstance(score, (int, float)):
                        top[1].metric("Score", f"{score:.4f}")
                    else:
                        top[1].metric("Score", str(score))
                    st.text_area("Content", value=content, height=220, key=f"result_content_{doc}_{idx}")

        elif status >= 400:
            st.error(data.get("error", f"Request failed (HTTP {status})."))

        st.button("Search again", use_container_width=True, on_click=reset_state, args=(["search_last_response", "search_query"],))

else:
    st.subheader("Ingest")

    ingest_type = st.radio(
        "Choose ingest method",
        ["Directory path", "Upload PDF(s)"],
        horizontal=True,
        key="ingest_type",
    )

    if ingest_type == "Directory path":
        st.info(
            "The directory path must exist on the **backend container** filesystem. "
            "If using Docker, mount a host folder into the backend and enter that mounted path (e.g. `/data`)."
        )
        dir_path = st.text_input("Directory path", placeholder="/data/uploads", key="ingest_dir_path")
        do_ingest = st.button("Ingest", type="primary", use_container_width=True)

        if do_ingest:
            p = (dir_path or "").strip()
            if not p:
                st.warning("Please enter a directory path.")
            else:
                with st.spinner("Ingesting…"):
                    status, payload = post_ingest_dir(p)
                payload = normalize_backend_error(payload)
                st.session_state["ingest_last_response"] = {"status": status, "data": payload}

    else:
        uploaded = st.file_uploader(
            "Drag & drop one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="ingest_files",
        )
        do_ingest = st.button("Ingest", type="primary", use_container_width=True)

        if do_ingest:
            if not uploaded:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Uploading & ingesting…"):
                    status, payload = post_ingest_files(uploaded)
                payload = normalize_backend_error(payload)
                st.session_state["ingest_last_response"] = {"status": status, "data": payload}

    resp = st.session_state.get("ingest_last_response")
    if resp:
        st.markdown("### Response")
        st.code(resp, language="json")

        status = resp["status"]
        data = resp["data"] if isinstance(resp["data"], dict) else {}

        if status == 200:
            st.success(data.get("message", "Ingested successfully."))
            files = data.get("files", [])
            if isinstance(files, list) and files:
                st.markdown("**Files:**")
                st.write(files)
        else:
            st.error(data.get("error", f"Ingest failed (HTTP {status})."))

        st.button(
            "Ingest again",
            use_container_width=True,
            on_click=reset_state,
            args=(["ingest_last_response", "ingest_dir_path", "ingest_files"],),
        )