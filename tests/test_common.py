import litstudy
import codecs

def test_robust_open():
    f = litstudy.common.robust_open
    expected = "ABC \U0001F600"

    assert f(b'').read() == ""

    content = expected.encode("utf8")
    assert f(content).read() == expected

    content = codecs.BOM_UTF8 + expected.encode("utf8")
    assert f(content).read() == expected

    content = codecs.BOM_UTF16_BE + expected.encode("utf_16_be")
    assert f(content).read() == expected

    content = codecs.BOM_UTF16_LE + expected.encode("utf_16_le")
    assert f(content).read() == expected

    # Contains some invalid UTF-8 character, should become U+FFFD
    content = b'ABC \x9f\x98\x80 DEF'
    assert f(content).read() == "ABC \ufffd\ufffd\ufffd DEF"
