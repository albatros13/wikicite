from __future__ import unicode_literals
import os
import lupa
from wikiciteparser.parser import parse_citation_dict, params_to_dict, is_citation_template_name, toPyDict


lua_data_path = os.path.dirname(__file__) + os.sep
print("Lua translator looking for data files in the following directory: ", lua_data_path)
# lua_data_path = os.path.join(os.path.dirname(__file__), 'cs1-translator-data.lua')

lua = lupa.LuaRuntime()
luacode = ''
lua_file_path = os.path.join(os.path.dirname(__file__), 'cs1-translator.lua')


with open(lua_file_path, 'r', encoding='utf-8') as f:
    luacode = f.read()


def translate_and_parse_citation_dict(arguments, template_name='citation'):
    split_template_name = template_name.strip().split(' ')
    template_name = split_template_name[-1] if len(split_template_name) > 1 else split_template_name[0]

    lua_table = lua.table_from(arguments)
    lua_result = lua.eval(luacode)(lua_table, 'it', template_name, lua_data_path)

    wrapped_type = lua.globals().type
    params = toPyDict(lua_result, wrapped_type)

    return parse_citation_dict(params, template_name)


def translate_and_parse_citation_template(template, lang='en'):
    name = template.name
    if not is_citation_template_name(name, lang):
        print("Not a citation template:", name)
        return {}
    return translate_and_parse_citation_dict(params_to_dict(template.params), template_name=name)
