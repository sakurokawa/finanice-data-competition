import typing

class Topic(typing.TypedDict):
        topic: str
        details: list[str]

class Discription(typing.TypedDict):
        company: str
        info:list[Topic]