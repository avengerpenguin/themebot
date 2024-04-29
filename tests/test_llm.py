from textwrap import dedent

from langchain_core.language_models import BaseLLM


def test_naming(llm: BaseLLM):
    print(
        llm.invoke(
            "Suggest a poetic or evocative name for the colour defined by hex code #E0E4CC"
        )
    )


def test_code(llm: BaseLLM):
    print(
        llm.invoke(
            "A designer has asked you to use the colour 'Giant Goldfish'; what hex code do you think best matches that?"
        )
    )


def test_palette(llm: BaseLLM):
    print(
        llm.invoke(
            dedent(
                """
        You are an expert at graphic design and colour themes.
        I will give you a list of colours as hex codes that together form a colour palette used in a design.
        I want you to come up with a creative name for this combined palette.

        The colours: #69D2E7,#A7DBD8,#E0E4CC,#F38630,#FA6900,#69D2E7,#A7DBD8,#E0E4CC

        Please respond with just a JSON object with a single key called "name" and the value of that key is your chosen name for the palette.
        """
            )
        )
    )


def test_scheme(llm: BaseLLM):
    print(
        llm.invoke(
            dedent(
                """
        You are an expert at graphic design and colour themes.

        I will provide you with a theme for a website and you should come up with a list of 5 colours for the website's theme.
        The colours should work nicely together to fit an overall website theme.

        There should be one colour for each of the following roles:

        - Background
        - Text
        - Primary (main CTAs and sections)
        - Secondary (less important buttons and info cards)
        - Accent (appears in images, highlights, hyperlinks, boxes, cards, etc.)

        Consider also that foreground colours should have a good contrast from text colour for accessibility.

        Give each colour as a hex code but also come up with a poetic or evocative name for it.

        Please respond with a JSON object with four properties.
        The first property is "name" where the value is a poetic or evocative name for the overall colour theme.
        The second property is "main_font" with a suggested font name/family to use as the default text
        The third property is "heading_font" with a suggested font for headings in this theme
        The fourth property is "colours" with the key being a JSON array of the colours with item being a JSON object with the following properties:

        - There is a key "code" where the value is the HTML hex code of the chosen colour
        - There is a key "name" where the value is the poetic or evocative name for that colour
        - There is a key "role" where the value is what role the colour plays in the overall theme, e.g. background, text, accent, primary, secondary

        The website theme is "mead".
        """
            )
        )
    )
