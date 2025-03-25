DPI = 300 
TITLE_FONT_SIZE = 10
AXIS_LABELS_FONT_SIZE = 9
AXIS_TICKS_FONT_SIZE = 7
BAR_LABEL_FONT_SIZE = 7


def validate_anndata(adata, required_columns):
    """
    Validates that the required columns exist in the AnnData object's `.obs`.

    Parameters:
        adata (AnnData): Input AnnData object.
        required_columns (list): List of columns required in `adata.obs`.

    Raises:
        ValueError: If any required column is missing.
    """
    for col in required_columns:
        if col not in adata.obs:
            raise ValueError(f"The column '{col}' does not exist in `adata.obs`.")

def validate_response_column(adata, response_col):
    """
    Validates and maps the response column in the AnnData object.

    Parameters:
        adata (AnnData): Input AnnData object.
        response_col (str): Column in `adata.obs` containing response labels.

    Returns:
        None
    """
    # Ensure response values are binary ('R', 'NR', 1, 0)
    valid_labels = {"R", "NR", 1, 0}
    unique_labels = set(adata.obs[response_col].unique())
    if not unique_labels.issubset(valid_labels):
        raise ValueError(
            f"The response column '{response_col}' contains invalid values: {unique_labels - valid_labels}. "
            f"Allowed values are {valid_labels}."
        )

    # Map 'R'/'NR' to 1/0 if necessary
    adata.obs[response_col] = adata.obs[response_col].map({"R": 1, "NR": 0, 1:1, 0:0})
    adata.obs[response_col] = adata.obs[response_col].astype(int)
