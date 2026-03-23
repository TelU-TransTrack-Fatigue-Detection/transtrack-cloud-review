from pydantic import BaseModel


class IncomingAlarm(BaseModel):
    id: str
    imei: str
    time: str
    alarm: str
    dms_video_url: str


class ReviewResult(BaseModel):
    id: str
    imei: str
    time: str
    alarm: str
    dms_video_url: str
    dms_video_url_after_proccess: str  # typo preserved — matches API contract verbatim
    confidence_level: int
    review_result: bool
    process_duration: int
    other: dict
