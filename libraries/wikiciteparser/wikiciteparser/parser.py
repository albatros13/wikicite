from __future__ import unicode_literals
import re
import os
import lupa
import mwparserfromhell
import importlib

lua = lupa.LuaRuntime()
lua_data_path = os.path.dirname(__file__) + os.sep
print("Lua translator looking for data files in the following directory: ", lua_data_path)

lua_parser_path = os.path.join(os.path.dirname(__file__), 'cs1.lua')
with open(lua_parser_path, 'r', encoding='utf-8') as f:
    lua_parser = f.read()

lua_translator_path = os.path.join(os.path.dirname(__file__), 'cs1-translator.lua')
with open(lua_translator_path, 'r', encoding='utf-8') as f:
    lua_translator = f.read()


# MediaWiki utilities simulated by Python wrappers
def lua_to_python_re(regex):
    rx = re.sub('%a', '[a-zA-Z]', regex) # letters
    rx = re.sub('%c', '[\x7f\x80]', regex) # control chars
    rx = re.sub('%d', '[0-9]', rx) # digits
    rx = re.sub('%l', '[a-z]', rx) # lowercase letters
    rx = re.sub('%p', '\\p{P}', rx) # punctuation chars
    rx = re.sub('%s', '\\s', rx) # space chars
    rx = re.sub('%u', '[A-Z]', rx) # uppercase chars
    rx = re.sub('%w', '\\w', rx) # alphanumeric chars
    rx = re.sub('%x', '[0-9A-F]', rx) # hexa chars
    return rx


def ustring_match(string, regex):
    return re.match(lua_to_python_re(regex), string) is not None


def ustring_len(string):
    return len(string)


def uri_encode(string):
    return string


def text_split(string, pattern):
    return lua.table_from(re.split(lua_to_python_re(pattern), string))


def nowiki(string):
    try:
        wikicode = mwparserfromhell.parse(string)
        return wikicode.strip_code()
    except (ValueError, mwparserfromhell.parser.ParserError):
        return string


# Conversion utilities, from lua objects to python objects
def is_int(val):
    """
    Is this lua object an integer?
    """
    try:
        x = int(val)
        return True
    except (ValueError, TypeError):
        return False


def to_py_dict(lua_val, wrapped_type):
    """
    Converts a lua dict to a Python one
    """
    wt = wrapped_type(lua_val)
    if wt == 'string':
        return nowiki(lua_val)
    elif wt == 'table':
        dct = {}
        lst = []
        for k, v in sorted(lua_val.items(), key=(lambda x: x[0])):
            vp = to_py_dict(v, wrapped_type)
            if not vp:
                continue
            if is_int(k):
                lst.append(vp)
            dct[k] = vp
        if lst:
            return lst
        return dct
    else:
        return lua_val


def parse_citation_dict(arguments, citation_type='citation'):
    """
    Parses the Wikipedia citation into a python dict.

    :param arguments: a dictionary with the arguments of the citation template
    :param citation_type: the name of the template used (e.g. 'cite journal', 'citation', and so on)
    :returns: a dictionary used as internal representation in wikipedia for rendering and export to other formats
    """

    if isinstance(arguments, dict):
        if 'CitationClass' not in arguments:
            arguments['CitationClass'] = citation_type
        if 'vauthors' in arguments:
            arguments['authors'] = arguments.pop('vauthors')
        if 'veditors' in arguments:
            arguments['editors'] = arguments.pop('veditors')
    else:
        if citation_type == "web" and len(arguments) > 0:
            res = {'CitationClass': 'web', 'url': arguments[0]}
            if len(arguments) > 1:
                res['Title'] = arguments[1]
            arguments = res
        else:
            print(citation_type, type(arguments), arguments)

    lua_table = lua.table_from(arguments)
    try:
        lua_result = lua.eval(lua_parser)(lua_table,
                                          ustring_match,
                                          ustring_len,
                                          uri_encode,
                                          text_split,
                                          nowiki)
    except Exception as e:
        print("Error in calling Lua code from parser: ", citation_type, arguments)
        return {'Title': 'Citation generic template not possible'}

    wrapped_type = lua.globals().type
    return to_py_dict(lua_result, wrapped_type)


def params_to_dict(params):
    """
    Converts the parameters of a mwparserfromhell template to a dictionary
    """
    dct = {}
    for param in params:
        dct[param.name.strip()] = param.value.strip()
    return dct


def translate_citation(arguments, citation_type, lang):
    """
    Translates citation to English
    """
    lua_table = lua.table_from(arguments)
    lua_result = lua.eval(lua_translator)(lua_table, lang, citation_type, lua_data_path)
    return to_py_dict(lua_result, lua.globals().type)


def is_citation_template_name(template_name, lang):
    lang_module = importlib.import_module('.' + lang, package='wikiciteparser')
    if template_name in lang_module.citation_template_names:
        return template_name


def parse_citation_template(template, lang='en'):
    """
    Takes a mwparserfromhell template object that represents
    a wikipedia citation, and converts it to a normalized representation
    as a dict.

    :returns: a dict representing the template, or None if the template
        provided does not represent a citation.
    """
    if not template.name:
        return {}

    template_name = template.name.strip().lower()
    split_template_name = template_name.split(' ')
    citation_type = split_template_name[-1] if len(split_template_name) > 1 else split_template_name[0]

    params = None
    if lang != 'en' and is_citation_template_name(template_name, lang):
        params = translate_citation(params_to_dict(template.params), citation_type, lang)
    elif is_citation_template_name(template_name, 'en'):
        params = params_to_dict(template.params)

    if params:
        return parse_citation_dict(params, citation_type)
    else:
        # print("Not a citation template:", lang, template_name)
        return {}

