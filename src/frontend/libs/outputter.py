import pandas as pd


TABLE_COLUMNS = {
    "video_id": "string",
    "frame_id": "string",
    "boi_id": "int",
    "temperatura °C": "int",
}

FEVER_THRESHOLD = 39


create_df = lambda: pd.DataFrame(
    {c: pd.Series(dtype=dt) for c, dt in TABLE_COLUMNS.items()}
)
create_df.__doc__ = """Returns: Pandas DataFrame('frame_id(string)', 'boi_id(int)', 'temperatura °C(int)')"""


append_row = lambda df, video_id, frame_id, temp: pd.concat(
    [
        df,
        pd.DataFrame(
            {
                "video_id": [video_id],
                "frame_id": [frame_id],
                "boi_id": [df["boi_id"].max() + 1] if not df.empty else 0,
                "temperatura °C": [temp],
                "febre": [temp >= FEVER_THRESHOLD],
            }
        ),
    ],
    ignore_index=True,
)
append_row.__doc__ = """
Parameters:
    df: DataFrame to append row to
    video_id: string indicating the video the frame comes from
    frame_id: string indicating frame id
    temp: int indicating the cattle's temperature in °C
Returns: pandas dataframe('frame_id(string)', 'boi_id(int)', 'temperatura °C(int)') with the appended row
"""

drop_row = lambda df, cow_id: df.drop(df[df.boi_id == cow_id].index)
drop_row.__doc__ = """
Parameters:
    df: DataFrame to drop row from
    cow_id: int indicating the cattle's id
returns: pandas dataframe('frame_id(string)', 'boi_id(int)', 'temperatura °c(int)') without the indicated row
"""


has_fever = lambda df, cow_id: (
    (
        True
        if df[df["boi_id"] == cow_id].iloc[0]["temperatura °C"] >= FEVER_THRESHOLD
        else False
    )
    if not df[df["boi_id"] == cow_id].empty
    else None
)
has_fever.__doc__ = f"""
Parameters:
    df: DataFrame to get the cattle from
    cow_id: int indicating the cattle's id 
returns: True if Cattle is above temperature threshold ({FEVER_THRESHOLD}), else False. If cattle was not found in DataFrame, None
"""

count_cattle_in_frame = lambda df, frame_id: df[df["frame_id"] == frame_id].shape[0]
count_cattle_in_frame.__doc__ = """
Parameters:
    df: DataFrame to get temperature from
    cow_id: int indicating the cattle's id 
returns: pandas dataframe('frame_id(string)', 'boi_id(int)', 'temperatura °c(int)') without the indicated row
"""
