class ConvertingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class EncodingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NoTextLayerError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ChaptersExtractingFailedWarning(Warning):
    def __init__(self, message):
        super().__init__(message)


class ParseImageWarning(Warning):
    def __init__(self, message):
        super().__init__(message)


class TitleExtractingWarning(Warning):
    def __init__(self, message):
        super().__init__(message)


class PageNumbersExtractingWarning(Warning):
    def __init__(self, message):
        super().__init__(message)


class FooterExtractingWarning(Warning):
    def __init__(self, message):
        super().__init__(message)
