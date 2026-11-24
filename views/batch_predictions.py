import streamlit as st

from src.dashboard import model_release
from src.predict import example_csv, export_csv, parse_batch, predict_batch
from src.schema import FEATURES, ID_COLUMN

st.title("Batch Predictions")
st.markdown("Upload a CSV to score multiple fictional customer profiles and download the results.")
bundle, _ = model_release()

with st.container(border=True, key="surface_batch_upload"):
    st.subheader("Upload customer profiles")
    st.caption(
        "Use fictional profiles. A file can contain up to 10,000 rows and must be 2 MB or smaller. "
        "Upload data and results are processed in this session."
    )
    upload = st.file_uploader(
        "Customer CSV", type="csv", key="batch_upload", on_change=lambda: st.session_state.pop("batch_result", None),
    )
    run, sample, download = st.columns(3)
    run_upload = run.button("Run batch predictions", type="primary", disabled=upload is None, width="stretch")
    run_sample = sample.button("Try the example", width="stretch")
    download.download_button(
        "Download example CSV", example_csv(), "churniq_example.csv", "text/csv",
        on_click="ignore", width="stretch",
    )

if run_upload or run_sample:
    st.session_state.pop("batch_result", None)
    try:
        content = upload.getvalue() if run_upload else example_csv()
        result = predict_batch(bundle, parse_batch(content))
        st.session_state["batch_result"] = (result, "Fictional example" if run_sample else "Uploaded CSV")
    except ValueError as error:
        st.error(str(error))

if "batch_result" in st.session_state:
    result, source = st.session_state["batch_result"]
    st.success(f"{source} · {len(result):,} customers scored.")
    st.subheader("Prediction results")
    st.dataframe(
        result.head(50), hide_index=True, width="stretch",
        column_config={
            ID_COLUMN: "Customer ID",
            "predicted_class": "Predicted churn",
            "churn_probability": st.column_config.NumberColumn("Churn probability", format="percent"),
        },
    )
    st.caption("The preview shows up to 50 rows. The download includes every scored row.")
    st.download_button(
        "Download prediction results", export_csv(result), "churniq_predictions.csv", "text/csv",
        type="primary", on_click="ignore",
    )

with st.expander("Required CSV columns"):
    st.code(",".join((ID_COLUMN, *FEATURES)), language=None)
    st.markdown(
        "Keep every required header and use a unique, nonempty customer ID per row. "
        "Blank feature values are allowed if the row has at least one value. "
        "Use the example CSV for category spelling and service combinations. "
        "Extra columns are ignored."
    )
