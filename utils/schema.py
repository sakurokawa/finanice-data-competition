import typing

class Topic(typing.TypedDict):
        topic: str
        details: list[str]

class Discription(typing.TypedDict):
        company: str
        info:list[Topic]

class Metadata_reportname(typing.TypedDict):
        title: str
        company: str

class Metadata_contents(typing.TypedDict):
        contents: str
        page: str