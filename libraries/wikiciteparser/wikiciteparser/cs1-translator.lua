--[[

TODO: |título=Copia archivada -> |title=Archive copy?  Other languages?

TODO: make sure that citation bot doesn't improperly rename translatable templates; see $fix_it currently at https://github.com/ms609/citation-bot/blob/master/Template.php#L85

]]

function (citation, lang, template_name, data_path)

    local data = dofile(data_path .. 'cs1-translator-data.lua')

    local params_main_t = data.params_main_t;
    local params_dates_t = data.params_dates_t;
    local params_misc_dates_t = data.params_misc_dates_t;
    local params_identifiers_t = data.params_identifiers_t;
    local params_language_t = data.params_language_t;


--[[--------------------------< _ M O N T H _ X L A T E >------------------------------------------------------

 TODO Translate months

]]

   local function _date_xlate (args_t)

--         local t = {};
--         local day, month, year;
--
--         if 'dump' == args_t[1] then													-- args_t[1] = 'dump' to dump <month_names_t> table;
--             return dump (month_data.month_names_t);
--         end
--         if not args_t[1] then return nil end
--         for i, pattern in ipairs (month_data.patterns) do									-- spin through the patterns table looking for a match
--             local c1, c2, c3;														-- captures in the 'pattern' from the pattern table go here
--
--             c1, c2, c3 = mw.ustring.match (mw.text.trim (args_t[1]), pattern[1]);	-- one or more captures set if source matches patterns[i][1])
--             if c1 then																-- c1 always set on match
--
--                 t = {
--                     [pattern[2] or 'x'] = c1,										-- fill the table of captures with the captures
--                     [pattern[3] or 'x'] = c2,										-- take index names from pattern table and assign sequential captures
--                     [pattern[4] or 'x'] = c3,										-- index name may be nil in pattern table so "or 'x'" spoofs a name for this index in this table
--                     };
--                 day = t.d or '';													-- translate table contents to named variables;
--                 month = string.lower (t.m or '');
-- --                 month = mw.ustring.lower (t.m or '');								-- absent table entries are nil so set unused parts to empty string; lowercase for indexing
--                 month = month_data.override_names[month] or month_data.month_names_t[month];	-- replace non-English name with English name from translation tables
--                 year= t.y or '';
--
--                 if month then
--                     local df = table.concat ({pattern[2], pattern[3], pattern[4]}, '');	-- extract date format from pattern table (pattern[2], pattern[3], pattern[4])
--
--                     if 'dmy' == df then												-- for dmy dates
--                         return table.concat ({day, month, year}, ' ');				-- assemble an English language dmy date
--                     elseif 'my' == df then											-- for month year dates
--                         return table.concat ({month, year}, ' ');					-- assemble an English language dmy date
--                     elseif 'mdy' == df then											-- for mdy dates
--                         return string.format ('%s %s, %s', month, day, year);		-- assemble an English language mdy date
--                     elseif 'm' == df then											-- must be month (only valid option remaining)
--                         return month;												-- none of the above, return the translated month;
--                     end
--                 end
--                 break;																-- and done; if here found pattern match but did not find non-English month name in <month_names_t>
--             end
--         end
        return args_t[1];															-- if here, couldn't translate so return the original date
   end


    --[[--------------------------< DUMP >--------------------------------------------------------------
    Convert table to string for testing
]]
    local function dump(o)
       if type(o) == 'table' then
          local s = '{ '
          for k,v in pairs(o) do
             if type(k) ~= 'number' then k = '"'..k..'"' end
             s = s .. '['..k..'] = ' .. dump(v) .. ','
          end
          return s .. '} '
       else
          return tostring(o)
       end
    end

    --[[--------------------------< I N _ A R R A Y >--------------------------------------------------------------
    Whether needle is in haystack
    ]]
    local function in_array (needle, haystack)
        if needle == nil then
            return false;
        end
        for n, v in ipairs (haystack) do
            if v == needle then
                return n;
            end
        end
        return false;
    end


    --[[--------------------------< A R G S _ G E T >--------------------------------------------------------------

get parameter names and values into an associative table from <frame> (the parameters in the #invoke) and from
the parent frame (the parameters in the calling template); set all parameter names (keys) to lower case, skip
parameters with empty-string values, skip parameters with whitespace-only values, skip duplicate parameter names
(parameter names in the #invoke frame have precedence over the same parameter name in the template frame).

This replaces Module:Arguments.getArgs() because this allows us to set parameter names to lowercase; something
that can't be done with getArgs()

returns <count> (the number of parameters added to <args_t>) because #args_t doesn't always work
]]
    local function args_get (frame, args_t)
        for k, v in pairs(frame) do										     -- for each parameter in the <frames_t.args> associative table
            if 'string' == type (k) then										 -- ignore positional parameters
                if v and not (args_t[k] or ('' == v) or (v:match ('^%s$'))) then -- skip when <args_t[k]> already present, skip when <v> is empty-string, skip when <v> is whitespace
                    args_t[k] = v;												 -- save k/v pair to in <args_t>
                end
            end
        end
    end


    --[[--------------------------< D A T E _ M A K E >------------------------------------------------------------

<args_t> - table of k/v pairs where k is the non-English parameter name and v is the assigned value from frame
<params_dates_t> - table of k/v_t pairs where k is the date part and v_t is a sequence table of non-English date-holding parameter names
<cite_args_t> - a sequence table that holds parameter=value strings (without pipes); for sorting

if args_t[<date_param>] is set, translate month name (if necessary) else assemble and translate date from date
parts args_t[<year_param>], args_t[<month_param>], args_t[<day_param>].  Resulting format is always dmy.  Unset
all date values after date created.  add translated date as text string to <cite_args_t> sequence table

month names in non-English |year= parameters that hold more than the year portion of a data (French |année=8 mars 2007 for example)
are NOT transla

]]

    local function date_make (args_t, cite_args_t, params_dates_t)
        local date, year, month, day;

        if params_dates_t.date_t then												-- TODO: is there a better way to do this?
            for _, v in ipairs (params_dates_t.date_t) do							-- loop through the date-holding parameters
                date = args_t[v];													-- will be nil if not set
                if date then break end												-- if set, we're done
            end
        end

        if params_dates_t.year_t then
            for _, v in ipairs (params_dates_t.year_t) do							-- loop through the year-holding parameters -- may also hold month and or day
                year = args_t[v];													-- will be nil if not set
                if year then break end												-- if set, we're done
            end
        end

        if params_dates_t.month_t then
            for _, v in ipairs (params_dates_t.month_t) do							-- loop through the month-holding parameters
                month = args_t[v];													-- will be nil if not set
                if month then break end												-- if set, we're done
            end
        end

        if params_dates_t.day_t then
            for _, v in ipairs (params_dates_t.day_t) do								-- loop through the day-holding parameters
                day = args_t[v];													-- will be nil if not set
                if day then break end												-- if set, we're done
            end
        end

        if date then
            date = _date_xlate ({date});											-- attempt translation

        elseif year then															-- if 'year'; without year, any spurious 'month' and/or 'day' params meaningless; pass on as-is to cs1|2 for error handling
            if month then
                month = _date_xlate ({month});										-- if there is a month parameter, translate its value
                local date_parts_t = {day, month, year};
                local date_t = {}
                for i=1, 3 do														-- done this way because members of <date_parts_t> might be nil
                    if date_parts_t[i] then											-- if not nil
                        table.insert (date_t, date_parts_t[i]);						-- add to a temporary table
                    end
                end

                date = table.concat (date_t, ' ');									-- make the dmy date string
            else
                --			date = year;														-- no date so make |date=<year>
                date = _date_xlate ({year});										-- attempt translation if non-English |year= has a month name
            end
            year = nil;																-- unset no longer needed
        end

        local keys_t = {'date_t', 'year_t', 'month_t', 'day_t'};

        for _, key in ipairs (keys_t) do											-- loop through the keys_t sequence table
            if params_dates_t[key] then												-- if there is a matching table
                for _, param in ipairs (params_dates_t[key]) do						-- get each parameter name and
                    args_t[param] = nil;											-- unset because no longer needed
                end
            end
        end

        if date then
            table.insert (cite_args_t, table.concat ({'date=', date}));				-- create and save parameter-like string (without pipe)
            if year then
                table.insert (cite_args_t, table.concat ({'year=', year}));			-- do the same here; year because both |date= and |year= are allowed in cs1|2
            end
        end
    end


    --[[--------------------------< M I S C _ D A T E _ M A K E >--------------------------------------------------

<args_t> - table of k/v pairs where k is the non-English parameter name and v is the assigned value from frame
<cite_args_t> - a sequence table that holds parameter=value strings (without pipes); for sorting
<in_lang> - language code index into the non-English parameter names table

TODO: translate |orig-date=?  can have a translatable date but can also have extraneous text ...  At this writing,
_date_xlate() expects only a date.

]]

    local function misc_date_make (args_t, cite_args_t, in_lang)
        local misc_date;

        for _, lang in ipairs ({in_lang, 'en'}) do									-- first do non-English names then, because they might be present, look for English parameter names
            for param, en_param in pairs (data.params_misc_dates_t[lang]) do
                if args_t[param] then												-- if the non-English parameter has a value
                    misc_date = _date_xlate ({args_t[param]});						-- translate the date
                    table.insert (cite_args_t, table.concat ({en_param, '=', misc_date}));	-- make the english parameter
                    args_t[param] = nil;											-- unset, consumed no longer needed
                end
            end
        end
    end


    --[[--------------------------< S E R I E S _ M A K E >--------------------------------------------------------

assemble the various 'series' parts into |series=
series={{{Reihe|}}} {{{NummerReihe|}}} {{{BandReihe|}}} {{{HrsgReihe|}}}

TODO: should this function be German only or does it need to allow other languages? is German the only language
that combines multiple elements into |series=?

]]

    local function series_make (args_t, cite_args_t)
        local series_t = {};

        local params = {'reihe', 'nummerreihe', 'bandreihe', 'hrsgreihe'};

        for _, param in ipairs (params) do
            if args_t[param] then
                table.insert (series_t, args_t[param]);								-- add to temp sequence table
                args_t[param] = nil;												-- unset, no longer needed
            end
        end

        if 0 ~= #series_t then
            local series = table.concat (series_t, ', ');							-- concatenate whichever parameters are present
            table.insert (cite_args_t, table.concat ({'series=', series}));			-- and make a parameter
        end
    end


    --[[--------------------------< I S X N _ M A K E >------------------------------------------------------------

make an |isbn= or |issn= parameter.  This function applies ((accept-as-written)) markup when there is some sort
of parameter equivalent to cs1|2's (now deprecated) |ignore-isbn-error=

]]

    local function isxn_make (args_t, cite_args_t, type, aliases_t, ignore_params_t, ignore_values_t)
        local isxn;
        local ignore_value;

        for _, v in ipairs (aliases_t) do											-- loop through the aliases_t sequence table
            if args_t[v] then
                isxn = args_t[v];
            end
            args_t[v] = nil;														-- unset because no longer needed
        end

        for _, v in ipairs (ignore_params_t) do										-- loop through the ignor_params_t sequence table
            if args_t[v] then
                ignore_value = args_t[v];
            end
            args_t[v] = nil;														-- unset because no longer needed
        end

        if isxn and ignore_value and (in_array (ignore_value, ignore_values_t) or ignore_values_t['*']) then	-- <ignore_values_t> is a table values that evaluate to 'yes' or wildcard
            table.insert (cite_args_t, table.concat ({type, '=((', isxn, '))'}));	-- make parameter but accept this isxn as written
        end
    end


    --[[--------------------------< A T _ M A K E >----------------------------------------------------------------

makes |at= for those templates that have things like |column=.  Not qualified by |page= or |pages= so that cs1|2
will emit an error message when both |at= and |page(s)= present.

<aliases_t> is a sequence table of non-English parameter names
<prefix> - text string that prefixes the value in <alias>; 'col.&nbsp;' for example

]]

    local function at_make (args_t, cite_args_t, aliases_t, prefix)
        for _, alias in ipairs (aliases_t) do
            if args_t[alias] then
                table.insert (cite_args_t, table.concat ({prefix, args_t[alias]}));
            end
            args_t[alias] = nil;													-- unset, no longer needed
        end
    end


    --[[--------------------------< C H A P T E R _ M A K E _ F R >------------------------------------------------
]]

    local function chapter_make_fr (args_t, cite_args_t)
        local chapter = args_t['titre chapitre'] or args_t['chapitre'];
        if chapter and (args_t['numéro chapitre'] or args_t['numéro']) then
            chapter = (args_t['numéro chapitre'] or args_t['numéro']) .. ' ' .. chapter;
            table.insert (cite_args_t, 'chapter=' .. chapter);
            args_t['titre chapitre'] = nil;											-- unset as no longer needed
            args_t['chapitre'] = nil;
            args_t['numéro chapitre'] = nil;
            args_t['numéro'] = nil;
        end
    end


    --[[--------------------------< I D _ M A K E >----------------------------------------------------------------

make a comma separated list of identifiers to be included in |id=

<params_identifiers_t> is a sequence table of sequence tables where:
    [1] is the non-English parameter name normalized to lower case
    [2] is the associated wikitext label to be used in the rendering
    [3] is the url prefix to be attached to the identifier value from the template parameter
    [4] is the url postfix to be attached to the identifier value

]]

    local function id_make (frame, args_t, cite_args_t, params_identifiers_t)
        local id_t = {};
        local value;

        for _, identifier_t in ipairs (params_identifiers_t) do
            if args_t[identifier_t[1]] then											-- if this identifier parameter has a value
                local value_t = {}
                if identifier_t[2] then												-- if there is a label (all except |id= should have a label)
                    table.insert (value_t, identifier_t[2]);						-- the label
                    table.insert (value_t, ':&nbsp;');								-- the necessary punctuation and spacing
                    if identifier_t[3] then											-- if an extlink prefix
                        table.insert (value_t, '[');								-- open extlink markup
                        table.insert (value_t, identifier_t[3]);					-- the link prefix
                        table.insert (value_t, args_t[identifier_t[1]]);			-- the identifier value
                        if identifier_t[4] then										-- postfix?
                            table.insert (value_t, identifier_t[4]);				-- the link postfix
                        end
                        table.insert (value_t, ' ');								-- require space between url and label
                        table.insert (value_t, args_t[identifier_t[1]]);			-- the identifier value as label
                        table.insert (value_t, ']');								-- close extlink markup
                    else
                        table.insert (value_t, args_t[identifier_t[1]]);			-- the identifier value
                    end
                else
                    table.insert (value_t, args_t[identifier_t[1]]);				-- no label so value only
                end

                table.insert (id_t, table.concat (value_t));						-- add to temp sequence table
                args_t[identifier_t[1]] = nil;										-- unset, no longer needed
            end
        end

        if 0 ~= #id_t then
            local id = table.concat (id_t, ', ');									-- concatenate whichever parameters are present into a comma-separated list
            table.insert (cite_args_t, table.concat ({'id=', id}));					-- and make a parameter
        end
    end


    --[[--------------------------< T I T L E _ M A K E _ F R >----------------------------------------------------

join |title= with |subtitle= to make new |title=

]]

    local function title_make_fr (args_t, cite_args_t)
        local title = args_t['titre'];												-- get the 'required' title parameter
        args_t['titre'] = nil;														-- unset as no longer needed
        if not title then
            title = args_t['titre original'] or args_t['titre vo'];					-- if |titre= empty or missing use one of these 'aliases'
            args_t['titre original'] = nil;											-- unset as no longer needed
            args_t['titre vo'] = nil;
        end

        if title then																-- only when there is a title
            if args_t['sous-titre'] then											-- add subtitle if present
                title = title .. ': ' .. args_t['sous-titre'];
                args_t['sous-titre'] = nil;											-- unset as no longer needed
            end
            table.insert (cite_args_t, 'title=' .. (title or ''));					-- add to cite_args_t
        end
    end


    --[[--------------------------< T I T L E _ M A K E _ P T >----------------------------------------------------

join |title= with |subtitle= to make new |title=

]]

    local function title_make_pt (args_t, cite_args_t)
        local title = args_t['título'] or args_t['titulo'] or args_t['titlo'];												-- get the 'required' title parameter
        args_t['título'] = nil;														-- unset as no longer needed
        args_t['titulo'] = nil;
        args_t['titlo'] = nil;

        if not title then
            return
        end

        if args_t['subtítulo'] or args_t['subtitulo'] then												-- add subtitle if present
            title = title .. ': ' .. (args_t['subtítulo'] or args_t['subtitulo']);
            args_t['subtítulo'] = nil;												-- unset as no longer needed
            args_t['subtitulo'] = nil;
        end
        table.insert (cite_args_t, 'title=' .. (title or ''));						-- add to cite_args_t
    end


    --[[--------------------------< U R L _ S T A T U S _ M A K E >------------------------------------------------

<aliases_t> is a sequence table of |dead-url= parameter aliases
<no_values_t> is a table of k/v pairs where k is a valid parameter value that means 'no' as in |dead-url=no (ie |url-status=live)

]]

    local function url_status_make (args_t, cite_args_t, aliases_t, no_values_t)
        for _, alias in ipairs (aliases_t) do										-- loop through the aliases_t sequence table
            if args_t[alias] and no_values_t[args_t[alias]] then					-- if the alias has a value and the value equates to 'no'
                table.insert (cite_args_t, 'url-status=live')
            end
            args_t[alias] = nil;													-- unset because no longer necessary
        end
    end


    --[[--------------------------< U R L _ A C C E S S _ M A K E >------------------------------------------------

<aliases_t> is a sequence table of |subscription= or |registration= parameter aliases
<values_t> is a table of k/v pairs where k is a valid parameter value that means 'yes' as in |subscription=yes (ie |url-access=subscription)

<values_t> can have a wildcard ('*'=true) which indicates that anything assigned to |subscription= or |registration= is sufficient

]]

    local function url_access_make (args_t, cite_args_t, type, aliases_t, values_t)

        for _, alias in ipairs (aliases_t) do										-- loop through the aliases_t sequence table
            if in_array (args_t[alias], values_t) or (args_t[alias] and values_t['*']) then	-- if the alias has a value and the value equates to 'yes' or the wildcard
                if 'subscription' == type then
                    table.insert (cite_args_t, 'url-access=subscription')
                else
                    table.insert (cite_args_t, 'url-access=registration')
                end
            end
            args_t[alias] = nil;													-- unset because no longer necessary
        end
    end


    --[[--------------------------< L A N G U A G E _ T A G _ G E T >----------------------------------------------

Test <lang> to see if it is a known language tag.  If it is, return <lang>.

When <lang> not a known language tag, search through <known_langs_t> for <lang> as a language name; if found
return the associated language tag; <lang> else.

]]

    local function language_tag_get (known_langs_t, lang)
        if mw.language.isKnownLanguageTag (lang) then
            return lang;															-- <lang> is a known language tag ('en', 'da', etc); return the tag
        end

        local lang_lc = lang:lower();												-- make lowercase copy for comparisons
        for tag, name in pairs (known_langs_t) do									-- loop through <known_langs_t>
            if lang_lc == name:lower() then											-- looking for <lang_lc>
                return tag;															-- found it, return the associated language tag
            end
        end

        return lang;																-- not a known language or tag; return as is
    end


    --[[--------------------------< L A N G U A G E _ M A K E >----------------------------------------------------

this function looks at <lang_param_val> to see if it is a language tag known by WikiMedia.  If it is a known
language tag, adds |language=<lang_param_val> to cite_args_t and unsets args_t[<lang_param>].

when <lang_param_val> is not a known language tag, fetches a k/v table of known language tags and names from MediaWiki
for the <in_lang> language (a language tag) where 'k' is a language tag and 'v' is the associated language name.

searches that table for <lang_param_val>.  If found, adds |language=<tag> to cite_args_t and unsets args_t[<lang_param>]

this function takes no action and returns nothing when <lang_param_val> is not known.

language names are expected to be properly capitalized and spelled according to the rules of the source wiki so
the match must be exact.

]]

    local function language_make (args_t, cite_args_t, in_lang, lang_params_t)
        local lang_param;
        local lang_param_val;

        for _, v in ipairs (lang_params_t[in_lang]) do								-- loop through the list of 'language' parameters supported at <in_lang>.wiki
            if args_t[v] then														-- if this parameter has a value
                lang_param = v;														-- save the parameter name
                lang_param_val = args_t[v];											-- save the parameter value
            end
            args_t[v] = nil;														-- unset not needed; if multiple 'language' parameters set, we use the 'last' one
        end

        if not lang_param_val then
            return;																	-- no 'language' parameter in this template so done
        end

        local known_langs_t = mw.language.fetchLanguageNames (in_lang, 'all');		-- get k/v table of language names from MediaWiki for <in_lang> language; (k/v pair is ['tag'] = 'name')
        local tag = language_tag_get (known_langs_t, lang_param_val);				-- returns valid language tag or content of <lang_param_val>
        table.insert (cite_args_t, table.concat ({'language=', tag}));				-- prefer the 'tag' because that is more translatable
        args_t[lang_param] = nil;													-- unset because no longer needed
    end


    --[[--------------------------< L A N G U A G E _ M A K E _ P T >----------------------------------------------

Special case for pt which supports some 'language' parameters that are not enumerated and some 'language' parameters
that are enumerated.

Inspects the non-enumerated parameters |codling=, |in=, and |ling=; if any set, return the value.

If none of the non-enumerated parameters are set, inspect the enumeratable parameters |idioma=, |língua=, |lingua=
in that order.

Collects the value from one of the non-enumerated parameters and returns that value or collects all of the values
from one set of the enumeratable parameters and returns only that set of languages.

If other non-enumerated parameters have values or other enumeratable parameter sets have values, they are ignored
so that the template will emit unrecognized parameter errors for the improper mixture of parameter names.

TODO: possible to share this with pl?

]]

    local function language_make_pt (args_t, cite_args_t)
        local known_langs_t = mw.language.fetchLanguageNames ('pt', 'all');			-- get a table of known language names and tags for Portuguese
        local language;

        for _, lang_param in ipairs ({'codling', 'in', 'ling'}) do					-- non-enumerated language parameters
            if args_t[lang_param] then
                language = args_t[lang_param];
                args_t[lang_param] = nil;
                return language_tag_get (known_langs_t, language);					-- attempt to get a language tag; return tag or <language>
            end
        end

        local langs_t = {};
        for _, lang_param in ipairs ({'idioma', 'língua', 'lingua'}) do				-- for each set of enumeratable language parameters
            local i=1;																-- make an enumerator
            while 1 do																-- loop forever
                if i == 1 then
                    if args_t[lang_param] or args_t[lang_param..i] then				-- if non-enumerated or its enumerated alias
                        table.insert (langs_t, (args_t[lang_param] or args_t[lang_param..i]));	-- prefer non-enumerated
                        args_t[lang_param] = nil;									-- unset as no longer needed
                        args_t[lang_param..i] = nil;

                    else
                        break;														-- no <lang_param> or <lang_param1> parameters; break out of while
                    end

                elseif args_t[lang_param..i] then
                    table.insert (langs_t, args_t[lang_param..i]);
                    args_t[lang_param..i] = nil;									-- unset as no longer needed

                elseif 0 ~= #langs_t then											-- here when <lang_param..(i-1)> but no <lang_param..i>
                    for i, lang in ipairs (langs_t) do								-- loop through <langs_t> and
                        langs_t[i] = language_tag_get (known_langs_t, lang);		-- attempt to get a language tag; returns tag or <lang>
                    end
                    return table.concat (langs_t, ', ');							-- make a string and done

                else
                    break;															-- no <lang_param .. i> parameter; break out of while; should not get here
                end
                i = i + 1;															-- bump the enumerator
            end
        end
    end


    --[[--------------------------< N A M E _ L I S T _ S T Y L E _ M A K E >--------------------------------------

<aliases_t> is a sequence table of |name-list-style= parameter aliases
<values_t> is a table of k/v pairs where 'k' is is the non-English parameter value and 'v' is its translation
            for parameters that are |last-author-amp= translations, in the function call write: {['*'] = 'amp'}

]]

    local function name_list_style_make (args_t, cite_args_t, aliases_t, values_t)
        for _, alias in ipairs (aliases_t) do										-- loop through the aliases_t sequence table
            if args_t[alias] and (values_t[args_t[alias]] or values_t['*']) then	-- if the alias has a recognized value
                table.insert (cite_args_t, table.concat ({'name-list-style=', values_t[args_t[alias]] or values_t['*']}));
            end
            args_t[alias] = nil;													-- unset because no longer necessary
        end
    end


    --[[--------------------------< R E N D E R >------------------------------------------------------------------

common renderer that translates parameters (after special case translations) and then renders the cs1|2 template

]]

    local function render (frame, args_t, cite_args_t, params_main_t, template)
        local out_t = {}															-- associative table for frame:expandTemplate
        local expand = args_t.expand;												-- save content of |expand= to render a nowiki'd version of the translated template
        args_t.expand = nil;														-- unset so we don't pass to cs1|2

        for param, val in pairs (args_t) do											-- for each parameter in the template
            if val and ('subst' ~= param) then										-- when it has an assigned value; skip '|subst=subst:' (from AnomieBOT when it substs the cs1 template)
                local enum = param:match ('%d+');									-- get the enumerator if <param> is enumerated; nil else
                local param_key = param:gsub ('%d+', '#');							-- replace enumerator with '#' if <param> is enumerated

                if params_main_t[param_key] then									-- if params_main_t[<param_key>] maps to a cs1|2 template parameter
                    local en_param = params_main_t[param_key];
                    if enum then
                        en_param = en_param:gsub ('#', enum);						-- replace '#' in enumerated cs1|2 parameter with enumerator from non-English parameter
                    end
                    table.insert (cite_args_t, table.concat ({en_param, '=', val}));	-- use the cs1|2 parameter with enumerator if present
                else
                    table.insert (cite_args_t, table.concat ({param, '=', val}));	-- use non-English param
                end
            end
        end

        if expand then																-- to see the translation as a raw template, create a nowiki'd version
            table.sort (cite_args_t);												-- sort so that the nowiki rendering is pretty
            local xlation = table.concat ({'{{', template, ' |', table.concat (cite_args_t, ' |'), '}}'});	-- the template string
            xlation = frame:callParserFunction ({name='#tag', args={'nowiki', xlation}});					-- apply nowiki
            return table.concat ({'<code>', xlation, '</code>'});					-- wrap in code tags and done
        end

        for _, arg in ipairs (cite_args_t) do										-- spin through the sequence table of parameters
            local param, val = arg:match ('^([^=]+)=(.+)');							-- split parameter string
            if nil == param then error (dump (cite_args_t)) end
            out_t[param] = val;														-- and put the results in the output table
        end

        local xlated_msg = '<!-- auto-translated by Module:CS1 translator -->';

        res = {}
        for k,v in pairs(out_t) do
            m = params_main_t[k];
            if m ~=nil then
                res[m] = v
            else
                res[k] = v;
            end
        end
        res["CitationClass"] = template
        return res;
    end


    --[[--------------------------< D A T E S _ A N D _ L A N G U A G E >------------------------------------------

date and language parameter functions utilized for almost all translations (pl and pt support enumerated language
parameters)

]]

    local function dates_and_language (args_t, cite_args_t, in_lang)
        date_make (args_t, cite_args_t, params_dates_t[in_lang]);					-- assemble and translate |date=
        -- 	misc_date_make (args_t, cite_args_t, in_lang);
        --  language_make (args_t, cite_args_t, in_lang, params_language_t);			-- translate language from German to appropriate language tag
    end


    --[[=========================<< C O M B I N E D _ F U N C T I O N S >>=========================================
    ]]

    --[[--------------------------< _ C I T E _ C A >--------------------------------------------------------------
Common function to implement:
    {{cite web/Catalan}} (ca:Plantilla:Ref-web)

]]

    local function _cite_ca (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values
        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'ca');								-- translate dates and language parameters

        return render (frame, args_t, cite_args_t, params_main_t.ca, template);		-- now go render the citation
    end


    --[[--------------------------< C I T E _ N E W S _ C A >------------------------------------------------------

entry point for {{cite news/Catalan}}

]]

    local function cite_news_ca (frame)
        return _cite_ca (frame, 'cite news');
    end


    --[[--------------------------< C I T E _ W E B _ C A >--------------------------------------------------------

entry point for {{cite web/Catalan}}

]]

    local function cite_web_ca (frame)
        return _cite_ca (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ D A >--------------------------------------------------------------
<includeonly>{{safesubst:<noinclude />#invoke:Sandbox/trappist_the_monk/Literatur|cite_journal_da}}</includeonly><noinclude>{{documentation}}</noinclude>
Common function to implement:
    {{cite book/Danish}} (da:Skabelon:Kilde bog)
    {{cite journal/Danish}} (da:Skabelon:Kilde )
    {{cite web/Danish}} (da:Skabelon:Kilde www)

]]

    local function _cite_da (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values
        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'da');								-- translate dates and language parameters

        url_status_make (args_t, cite_args_t, {'dead-url', 'deadurl', 'dødtlink', 'dødlenke', 'deadlink', 'død-lenke'}, {['no'] = true, ['nej'] = true});

        url_access_make (args_t, cite_args_t, 'subscription', {'subscription', 'abonnement'}, {'yes', 'true', 'y', 'ja', 'sand', 'j'});
        url_access_make (args_t, cite_args_t, 'registration', {'registration'}, {'yes', 'true', 'y', 'ja', 'sand', 'j'});

        isxn_make (args_t, cite_args_t, 'isbn', {'isbn', 'isbn13'}, {'ignorer-isbn-fejl', 'ignoreisbnerror', 'ignore-isbn-error'}, {'yes', 'true', 'y', 'ja', 'sand', 'j'});

        return render (frame, args_t, cite_args_t, params_main_t.da, template);		-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ D A >------------------------------------------------------

entry point for {{cite book/Danish}}

]]

    local function cite_book_da (frame)
        return _cite_da (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ D A >------------------------------------------------

entry point for {{cite journal/Danish}}

]]

    local function cite_journal_da (frame)
        return _cite_da (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ D A >--------------------------------------------------------

entry point for {{cite web/Danish}}

This function called by intermediate function cite_web_da_no() because da.wiki and no.wiki both use {{kilde www}}

]]

    local function cite_web_da (frame)
        return _cite_da (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ D E >--------------------------------------------------------------

implements {{Literatur}} (de:Vorlage:Literatur), {{Cite web/German}} (de:Vorlage:Internetquelle)

]]

    local function _cite_de (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'de');								-- translate dates and language parameters

        series_make (args_t, cite_args_t);											-- assemble |series=

        local isxn;

        for _, param in ipairs ({'isbnformalfalsch', 'isbndefekt'}) do				-- these two parameters take a 'broken' but valid isbn
            if args_t[param] then
                isxn = table.concat ({'isbn=((', args_t[param], '))'});				-- create |isbn=((<broken isbn>))
            end
            args_t[param] = nil;													-- unset because no longer needed
        end
        if isxn then
            table.insert (cite_args_t, isxn);										-- save the parameter
        end

        if args_t['issnformalfalsch'] then											-- these parameter takes a 'broken' but valid issn
            isxn = table.concat ({'issn=((', args_t['issnformalfalsch'], '))'});	-- create |issn=((<broken issn>))
            table.insert (cite_args_t, isxn);										-- save the parameter
        end
        args_t['issnformalfalsch'] = nil;											-- unset because no longer needed

        at_make (args_t, cite_args_t, {'Spalten'}, 'col.&nbsp;');					-- assemble |at=col. ...

        id_make (frame, args_t, cite_args_t, params_identifiers_t.de);						-- assemble |id=

        if args_t.hrsg then
            if template == 'cite web' then									-- cite web/German has different meaning from in de:Vorlage:Literatur (cite book/German)
                table.insert (cite_args_t, table.concat ({'publisher=', args_t.hrsg}));
            else																	-- different meaning from the meaning of this same parameter in de:Vorlage:Internetquelle (cite web/German)
                table.insert (cite_args_t, table.concat ({'editor=', args_t.hrsg}));
            end
            args_t.hrsg = nil;														-- unset, no longer needed
        end

        if args_t.offline then
            args_t.offline = nil;													-- any value means |url-status=dead; there is no form of this param that means |url-status=live; so just unset
        end

        return render (frame, args_t, cite_args_t, params_main_t.de, template);	-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ D E >------------------------------------------------------

entry point for {{cite book/German}}

]]

    local function cite_book_de (frame)
        return _cite_de (frame, 'citation');									-- TODO: change this to 'cite book/German'?
    end


    --[[--------------------------< C I T E _ W E B _ D E >--------------------------------------------------------

entry point for {{cite web/German}}

]]

    local function cite_web_de (frame)
        return _cite_de (frame, 'cite web');									-- TODO: change this to 'cite book/German'?
    end


    --[[--------------------------< _ C I T E _ E S >--------------------------------------------------------------

implements {{Cita libro}} (:es:Plantilla:Cita_libro)

]]

    local function _cite_es (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'es');								-- translate dates and language parameters

        url_status_make (args_t, cite_args_t, {'urlmuerta'}, {['no'] = true});
        url_access_make (args_t, cite_args_t, 'subscription', {'suscripción', 'subscription'}, {['*'] = true});
        url_access_make (args_t, cite_args_t, 'registration', {'registration', 'requiere-registro', 'requiereregistro', 'registro'}, {['*'] = true})

        isxn_make (args_t, cite_args_t, 'isbn', {'isbn', 'isbn13'}, {'ignore-isbn-error', 'ignoreisbnerror'}, {['*'] = true});	-- |ignore-isbn-error=<anything>

        name_list_style_make (args_t, cite_args_t, {'ampersand', 'lastauthoramp', 'last-author-amp'}, {['*']='amp'});

        return render (frame, args_t, cite_args_t, params_main_t.es, template);	-- now go render the citation

    end


    --[[--------------------------< C I T E _ B O O K _ E S >------------------------------------------------------

entry point for {{cita book}} (Spanish: {{cita libro}})

]]

    local function cite_book_es (frame)
        return _cite_es (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ E S >------------------------------------------------

entry point for {{cita publicación}} (es:Plantilla:Cita publicación)

]]

    local function cite_journal_es (frame)
        return _cite_es (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ N E W S _ E S >------------------------------------------------------

entry point for {{cita news}} (Spanish: {{cita noticia}})

]]

    local function cite_news_es (frame)
        return _cite_es (frame, 'cite news');
    end


    --[[--------------------------< C I T E _ W E B _ E S >--------------------------------------------------------

entry point for {{cita web}} (Spanish: {{cita web}})

]]

    local function cite_web_es (frame)
        return _cite_es (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ F I >--------------------------------------------------------------

implements {{Kirjaviite}} :fi:Malline:Kirjaviite

]]

    local function _cite_fi (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'fi');								-- translate dates and language parameters

        at_make (args_t, cite_args_t, {'palsta', 'palstat'}, 'col.&nbsp;')

        return render (frame, args_t, cite_args_t, params_main_t.fi, template);	-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ F I >------------------------------------------------------

entry point for {{cite book/Finnish}}

]]

    local function cite_book_fi (frame)
        return _cite_fi (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ F I >------------------------------------------------

entry point for {{cite journal/Finnish}}

]]

    local function cite_journal_fi (frame)
        return _cite_fi (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ F I >--------------------------------------------------------

entry point for {{cite web/Finnish}}

]]

    local function cite_web_fi (frame)
        return _cite_fi (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ F R >--------------------------------------------------------------

implements {{cite book/French}} (:fr:Modèle:Ouvrage)

]]

    local function _cite_fr (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'fr');								-- translate dates and language parameters

        if template:find ('book', 1, true) then
            chapter_make_fr (args_t, cite_args_t)
        else
            if args_t['numéro'] then
                cite_args_t.issue = args_t['numéro'];
                args_t['numéro'] = nil;
            end
        end

        title_make_fr (args_t, cite_args_t);

        if args_t['accès url'] then
            local values = {['inscription'] = 'subscription', ['payant'] = 'subscription', ['limité'] = 'limited', ['libre'] = 'libre'};	-- 'libre' not supported; include it to get the error message
            table.insert (cite_args_t, 'url-access=' .. (values[args_t['accès url']] or args_t['accès url']));
            args_t['accès url'] = nil;
        end

        url_status_make (args_t, cite_args_t, {'dead-url'}, {['no'] = true, ['non'] = true});

        if 'oui' == args_t['et al.'] or 'oui' == args_t['et alii'] then				-- accepted value 'oui'; special case |display-authors=etal
            table.insert (cite_args_t, 'display-authors=etal');
            args_t['et al.'] = nil;													-- unset as no longer needed
            args_t['et alii'] = nil;
        end

        if args_t['isbn erroné'] then
            table.insert (cite_args_t, 'isbn=((' .. args_t['isbn erroné'] .. '))');	-- apply accept-as-written markup
            args_t['isbn'] = nil;													-- can't have |isbn= and |isbn erroné=
            args_t['isbn erroné'] = nil;											-- unset as no longer needed
        end

        local volume;
        if args_t['titre volume'] or args_t['tome'] then
            if args_t['tome'] then
                volume = args_t['tome'];											-- begin with volume 'number'
            end

            if volume then
                volume = volume .. ' ' .. args_t['tome'];							-- append volume 'title'
            else
                volume = args_t['titre volume'];									-- just volume 'title'
            end
            args_t['titre volume'] = nil;											-- unset as no longer needed
            args_t['tome'] = nil;
        end

        id_make (frame, args_t, cite_args_t, params_identifiers_t.fr);						-- assemble |id=

        if args_t['pages'] then
            args_t['pages'] = nil;													-- unset; alias of |pages totales=; no equivalent in cs1|2
        end

        return render (frame, args_t, cite_args_t, params_main_t.fr, template);		-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ F R >------------------------------------------------------

entry point for {{cite book/French}}

]]

    local function cite_book_fr (frame)
        return _cite_fr (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ F R >------------------------------------------------

entry point for {{cite journal/French}}

]]

    local function cite_journal_fr (frame)
        return _cite_fr (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ F R >--------------------------------------------------------

entry point for {{cite web/French}}

]]

    local function cite_web_fr (frame)
        return _cite_fr (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ I T >--------------------------------------------------------------

implements {{Cita libro}} (:it:Template:Cita_libro)

]]

    local function _cite_it (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        dates_and_language (args_t, cite_args_t, 'it');								-- translate dates and language parameters

        url_status_make (args_t, cite_args_t, {'urlmorto'}, {['no'] = true});
        url_access_make (args_t, cite_args_t, 'subscription', {'richiestasottoscrizione'}, {['*'] = true});	-- |richiestasottoscrizione=<anything> -> |url-access=subscription

        isxn_make (args_t, cite_args_t, 'isbn', {'isbn', 'isbn13'}, {'ignoraisbn'}, {['*'] = true});	-- |ignore-isbn-error=<anything>

        name_list_style_make (args_t, cite_args_t, {'lastauthoramp'}, {['*']='amp'});	-- listed in ~/Whitelist and ~/Configuration but not supported in main module

        if args_t['etal'] then														-- apparently any value (typically 's', 'sì', or 'si'); rather like |display-authors=etal
            table.insert (cite_args_t, 'display-authors=etal');
            args_t['etal'] = nil;
        end
        if args_t['etalcuratori'] then
            table.insert (cite_args_t, 'display-editors=etal');
            args_t['etalcuratori'] = nil;
        end
        return render (frame, args_t, cite_args_t, params_main_t.it, template);	-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ I T >------------------------------------------------------

entry point for {{cite book/Italian}} it:Template:Cita libro

]]

    local function cite_book_it (frame)
        return _cite_it (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ I T >------------------------------------------------

entry point for {{cite journal/Italian}} it:Template:Cita pubblicazione

]]

    local function cite_journal_it (frame)
        return _cite_it (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ N E W S _ I T >------------------------------------------------------

entry point for {{cite news/Italian}} it:Template:Cita news

]]

    local function cite_news_it (frame)
        return _cite_it (frame, 'cite news');
    end


    --[[--------------------------< C I T E _ W E B _ I T >--------------------------------------------------------

entry point for {{cite web/Italian}} it:Template:Cita web

]]

    local function cite_web_it (frame)
        return _cite_it (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ N L >--------------------------------------------------------------
<includeonly>{{safesubst:<noinclude />#invoke:Sandbox/trappist_the_monk/Literatur|cite_journal_nl}}</includeonly><noinclude>{{documentation}}</noinclude>
Common function to implement:
    {{cite book/Dutch}} (nl:Sjabloon:Citeer boek)
    {{cite journal/Dutch}} (nl:Sjabloon:Citeer journal)
    {{cite web/Dutch}} (nl:Sjabloon:Citeer)

]]

    local function _cite_nl (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'nl');								-- translate dates and language parameters

        url_status_make (args_t, cite_args_t, {'deadurl', 'dead-url', 'dodeurl', 'dode-url'}, {['no'] = true, ['nee'] = true});

        id_make (frame, args_t, cite_args_t, params_identifiers_t.nl);					-- assemble |id=


        return render (frame, args_t, cite_args_t, params_main_t.nl, template);		-- now go render the citation
    end


    --[[--------------------------< C I T A T I O N _ N L >--------------------------------------------------------

entry point for {{citation/Dutch}}

]]

    --local function cite_book_nl (frame)
    --	return _cite_nl (frame, 'citation');
    --end


    --[[--------------------------< C I T E _ B O O K _ N L >------------------------------------------------------

entry point for {{cite book/Dutch}}

]]

    local function cite_book_nl (frame)
        return _cite_nl (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ N L >------------------------------------------------

entry point for {{cite journal/Dutch}}

]]

    local function cite_journal_nl (frame)
        return _cite_nl (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ N L >--------------------------------------------------------

entry point for {{cite web/Dutch}}

]]

    local function cite_web_nl (frame)
        return _cite_nl (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ P L >--------------------------------------------------------------
<includeonly>{{safesubst:<noinclude />#invoke:Sandbox/trappist_the_monk/Literatur|cite_journal_pl}}</includeonly><noinclude>{{documentation}}</noinclude>
Common function to implement:
    {{citation/Polish}} (pl:Szablon:Cytuj)
    {{cite book/Polish}} (pl:Szablon:Cytuj książkę)
    {{cite journal/Polish}} (pl:Szablon:Cytuj pismo)
    {{cite web/Polish}} (pl:Szablon:Cytuj stronę)

]]

    local function _cite_pl (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        -- 	date_make (args_t, cite_args_t, params_dates_t.pl);							-- assemble and translate |date=
        -- 	misc_date_make (args_t, cite_args_t, 'pl');

        if 'tak' == args_t['odn'] then												-- |ref=tak is more-or-less the same as |ref=harv at en.wiki
            args_t['odn'] = nil;													-- unset because superfluous at en.wiki
        end

        if args_t['tom'] or args_t['tytuł tomu'] then
            local volume;
            if args_t['tom'] then
                volume = args_t['tom'];
            end
            if args_t['tytuł tomu'] then
                if volume then
                    volume = volume .. ' ' .. args_t['tytuł tomu'];
                else
                    volume = args_t['tytuł tomu'];
                end
            end
            if volume then
                table.insert (cite_args_t, 'volume=' .. volume);
                args_t['tom'] = nil;												-- unset as no longer needed
                args_t['tytuł tomu'] = nil;
            end
        end

        local i=1;																	-- make a counter
        local language;
        while 1 do																	-- loop forever TODO: same enough as pt to combine with language_make_pt()?
            if i == 1 then
                if args_t['język'] or args_t['język1'] then							-- if non-enumerated or its enumerated alias
                    language = args_t['język'] or args_t['język1'];					-- prefer non-enumerated
                    args_t['język'] = nil;											-- unset as no longer needed
                    args_t['język1'] = nil;
                else
                    break;
                end
            elseif args_t['język'..i] then
                language = language .. ', ' .. args_t['język'..i];
                args_t['język'..i] = nil;											-- unset as no longer needed
            else
                break;
            end
            i = i + 1;																-- bump the counter
        end

        if language then
            table.insert (cite_args_t, 'language=' .. language);
        end

        return render (frame, args_t, cite_args_t, params_main_t.pl, template);		-- now go render the citation
    end


    --[[--------------------------< C I T A T I O N _ P L >--------------------------------------------------------

entry point for {{citation/Polish}}

]]

    local function citation_pl (frame)
        return _cite_pl (frame, 'citation');
    end


    --[[--------------------------< C I T E _ B O O K _ P L >------------------------------------------------------

entry point for {{cite book/Polish}}

]]

    local function cite_book_pl (frame)
        return _cite_pl (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ P L >------------------------------------------------

entry point for {{cite journal/Polish}}

]]

    local function cite_journal_pl (frame)
        return _cite_pl (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ P L >--------------------------------------------------------

entry point for {{cite web/Polish}}

]]

    local function cite_web_pl (frame)
        return _cite_pl (frame, 'cite web');
    end



    --[[--------------------------< _ C I T E _ P T >--------------------------------------------------------------

Common function to implement:
    {{citation/Portuguese}} (pt:Predefinição:Citation)
    {{cite book/Portuguese}} (pt:Predefinição:Citar livro)
    {{cite journal/Portuguese}} (pt:Predefinição:Citar periódico)
    {{cite news/Portuguese}} (pt:Predefinição:Citar jornal)
    {{cite web/Portuguese}} (pt:Predefinição:Citar web)

]]

    local function _cite_pt (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        -- 	date_make (args_t, cite_args_t, params_dates_t.pt);							-- assemble and translate |date=
        -- 	misc_date_make (args_t, cite_args_t, 'pt');

        title_make_pt (args_t, cite_args_t);										-- join |title= with |subtitle= to make new title

        url_status_make (args_t, cite_args_t, {'datali', 'dead-url', 'deadurl', 'li', 'ligação inactiva', 'ligação inativa', 'urlmorta'}, {['no'] = true, ['não'] = true});

--         local language = language_make_pt (args_t, cite_args_t);					-- get string of language tags and/or unknown language names (or nil if no language parameters)
--
--         if language then
--             table.insert (cite_args_t, 'language=' .. language);
--         end

        return render (frame, args_t, cite_args_t, params_main_t.pt, template);		-- now go render the citation
    end


    --[[--------------------------< C I T A T I O N _ P T >--------------------------------------------------------

entry point for {{citation/Portuguese}}

]]

    local function cite_book_pt (frame)
        return _cite_pt (frame, 'citation');
    end


    --[[--------------------------< C I T E _ B O O K _ P T >------------------------------------------------------

entry point for {{cite book/Portuguese}}

]]

    local function cite_book_pt (frame)
        return _cite_pt (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ P T >------------------------------------------------

entry point for {{cite journal/Portuguese}}

]]

    local function cite_journal_pt (frame)
        return _cite_pt (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ N E W S _ P T >------------------------------------------------------

entry point for {{cite news/Portuguese}}

]]

    local function cite_news_pt (frame)
        return _cite_pt (frame, 'cite news');
    end


    --[[--------------------------< C I T E _ W E B _ P T >--------------------------------------------------------

entry point for {{cite web/Portuguese}}

]]

    local function cite_web_pt (frame)
        return _cite_pt (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ S V >--------------------------------------------------------------
<includeonly>{{safesubst:<noinclude />#invoke:Sandbox/trappist_the_monk/Literatur|cite_journal_sv}}</includeonly><noinclude>{{documentation}}</noinclude>
Common function to implement:
    {{cite book/Swedish}} (sv:Mall:Bokref)
    {{cite journal/Swedish}} (sv:Mall:Tidskriftsref)
    {{cite web/Swedish}} (sv:Mall:Webbref)

]]

    local function _cite_sv (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'sv');								-- translate dates and language parameters

        name_list_style_make (args_t, cite_args_t, {'författarsep'}, {['*'] = 'amp'});

        id_make (frame, args_t, cite_args_t, params_identifiers_t.sv)

        return render (frame, args_t, cite_args_t, params_main_t.sv, template);		-- now go render the citation
    end


    --[[--------------------------< C I T A T I O N _ S V >--------------------------------------------------------

entry point for {{citation/Swedish}}

]]

    local function citation_sv (frame)
    	return _cite_sv (frame, 'citation');
    end


    --[[--------------------------< C I T E _ B O O K _ S V >------------------------------------------------------

entry point for {{cite book/Swedish}}

]]

    local function cite_book_sv (frame)
        return _cite_sv (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ S V >------------------------------------------------

entry point for {{cite journal/Swedish}}

]]

    local function cite_journal_sv (frame)
        return _cite_sv (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ S V >--------------------------------------------------------

entry point for {{cite web/Swedish}}

]]

    local function cite_web_sv (frame)
        return _cite_sv (frame, 'cite web');
    end


    --[[=========================<< C I T E   B O O K   F U N C T I O N S >>=======================================
    ]]

    --[[--------------------------< C I T E _ B O O K _ R U >------------------------------------------------------

implements {{Книга}} (:ru:Шаблон:Книга)

]]

    local function cite_book_ru (frame)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'ru');								-- translate dates and language parameters

        at_make (args_t, cite_args_t, {'столбцы'}, 'col.&nbsp;')

        return render (frame, args_t, cite_args_t, params_main_t.ru, 'cite book');	-- now go render the citation
    end


    local function cite_journal_ru (frame)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        dates_and_language (args_t, cite_args_t, 'ru');								-- translate dates and language parameters

        return render (frame, args_t, cite_args_t, params_main_t.ru, 'cite journal');	-- now go render the citation
    end

    local function cite_media_ru (frame)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        dates_and_language (args_t, cite_args_t, 'ru');								-- translate dates and language parameters

        return render (frame, args_t, cite_args_t, params_main_t.ru, 'cite AV media');	-- now go render the citation
    end

    local function cite_web_ru (frame)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        dates_and_language (args_t, cite_args_t, 'ru');								-- translate dates and language parameters

        return render (frame, args_t, cite_args_t, params_main_t.ru, 'cite web');	-- now go render the citation
    end

    --[[=========================<< C I T E   W E B   F U N C T I O N S >>=========================================
    ]]
    --[[--------------------------< _ C I T E _ N O >--------------------------------------------------------------

implements
    {{cite book/Norwegian}} (:no:Mal:Kilde bok)
    {{cite journal/Norwegian}} (:no:Mal:Kilde artikkel)
    {{cite web/Norwegian}} (:no:Mal:Kilde www)

]]

    local function _cite_no (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values
        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'no');								-- translate dates and language parameters

        url_status_make (args_t, cite_args_t, {'død-lenke', 'dødlenke'}, {['no'] = true, ['nei'] = true});
        url_access_make (args_t, cite_args_t, 'subscription', {'abonnement', 'abb'}, {'yes', 'true', 'y', 'ja'});	-- for |subscription=
        url_access_make (args_t, cite_args_t, 'registration', {'registrering'}, {'yes', 'true', 'y', 'ja'});		-- for |registration=

        if args_t['url-tilgang'] then												-- translate value assigned to |url-access= TODO: make this into a shared function?
            local values = {['abonnement'] = 'subscription', ['registrering'] = 'registration', ['begrenset'] = 'limited', ['åpen'] = 'åpen'};	-- 'åpen' not supported; include it to get the error message
            table.insert (cite_args_t, 'url-access=' .. (values[args_t['url-tilgang']] or args_t['url-tilgang']));
            args_t['url-tilgang'] = nil;
        end

        isxn_make (args_t, cite_args_t, 'isbn', {'isbn', 'isbn13'}, {'ignorer-isbn-feil', 'ignorerisbnfeil'}, {'yes', 'true', 'y', 'ja'});

        if not args_t['navnelisteformat'] then
            name_list_style_make (args_t, cite_args_t, {'sisteforfatteramp'}, {['*']='amp'});
        end

        return render (frame, args_t, cite_args_t, params_main_t.no, template);		-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ N O >------------------------------------------------------

entry point for {{cite web/Norwegian}}

]]

    local function cite_book_no (frame)
        return _cite_no (frame, 'cite book');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ N O >------------------------------------------------

entry point for {{cite journal/Norwegian}}

]]

    local function cite_journal_no (frame)
        return _cite_no (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ N E W S _ N O >------------------------------------------------------

entry point for {{cite news/Norwegian}}

]]

    local function cite_news_no (frame)
        return _cite_no (frame, 'cite news');
    end


    --[[--------------------------< C I T E _ W E B _ N O >--------------------------------------------------------

entry point for {{cite web/Norwegian}}

This function called by intermediate function cite_web_da_no() because da.wiki and no.wiki both use {{kilde www}}

]]

    local function cite_web_no (frame)
        return _cite_no (frame, 'cite web');
    end


    --[[--------------------------< _ C I T E _ T R >--------------------------------------------------------------

implements {{Web kaynağı}} (:tr:Şablon:Web_kaynağı, :tr:Şablon:Akademik dergi kaynağı (journal))

]]

    local function _cite_tr (frame, template)
        local args_t = {};															-- a table of k/v pairs that holds the template's parameters
        args_get (frame, args_t);													-- get the parameters and their values

        local cite_args_t = {};														-- a sequence table that holds parameter=value strings (without pipes); for sorting

        -- special cases
        dates_and_language (args_t, cite_args_t, 'tr');								-- translate dates and language parameters

        url_status_make (args_t, cite_args_t, {'ölüurl', 'ölü-url', 'bozukurl'}, {['hayır'] = true, ['h'] = true, ['no'] = true});
        if not (args_t['url-access'] or args_t['url-erişimi'] or args_t['URLerişimi']) then
            url_access_make (args_t, cite_args_t, 'subscription', {'subscription'}, {'yes', 'true', 'y', 'e', 'evet', 'doğru'})
            url_access_make (args_t, cite_args_t, 'registration', {'registration'}, {'yes', 'true', 'y', 'e', 'evet', 'doğru'})
        end

        isxn_make (args_t, cite_args_t, 'isbn', {'isbn', 'isbn13'}, {'ignore-isbn-error', 'ignoreisbnerror'}, {'yes', 'true', 'y', 'e', 'evet', 'doğru'});

        name_list_style_make (args_t, cite_args_t, {'last-author-amp', 'lastauthoramp', 'sonyazarve'}, {['*']='amp'});	-- listed in ~/Whitelist and ~/Configuration (SonYazarVe) not supported in main

        return render (frame, args_t, cite_args_t, params_main_t.tr, template);		-- now go render the citation
    end


    --[[--------------------------< C I T E _ B O O K _ T R >------------------------------------------------------

implements {{Web kaynağı}} (:tr:Şablon:Web_kaynağı)

]]

    local function cite_book_tr (frame)
        return _cite_tr (frame, 'cite book');
    end


-- entry point for {{cite news/Turkish}} (:tr:Şablon:Haber kaynağı)
    local function cite_news_tr (frame)
        return _cite_tr (frame, 'cite news');
    end


    --[[--------------------------< C I T E _ J O U R N A L _ T R >------------------------------------------------

implements {{Akademik dergi kaynağı}} (:tr:Şablon:Akademik dergi kaynağı)

]]

    local function cite_journal_tr (frame)
        return _cite_tr (frame, 'cite journal');
    end


    --[[--------------------------< C I T E _ W E B _ T R >--------------------------------------------------------

implements {{Web kaynağı}} (:tr:Şablon:Web_kaynağı)

]]

    local function cite_web_tr (frame)
        return _cite_tr (frame, 'cite web');
    end


    -- Main: choose a translation method
    -- Template names in unicode must be translated to English alternatives before calling translator
    local methods = {

-- Catalan
        news_ca = cite_news_ca,         		    							-- ca:Plantilla:Ref-publicació
        web_ca = cite_web_ca,			        								-- ca:Plantilla:Ref-web

-- Danish
        bog_da = cite_book_da,												    -- da:Skabelon:Kilde bog
        web_da = cite_web_da,													-- da:Skabelon:Kilde www or da:Skabelon:Cite web
        www_da = cite_web_da,													-- da:Skabelon:Kilde www or da:Skabelon:Cite web
        tidsskrift_da = cite_journal_da,										-- da:Skabelon:Kilde tidsskrift / da:Skabelon:Kilde artikel (da.wiki prefers da:Skabelon:cite journal)
        artikel_da = cite_journal_da,											-- da:Skabelon:Kilde tidsskrift / da:Skabelon:Kilde artikel (da.wiki prefers da:Skabelon:cite journal)

-- Dutch
        boek_nl = cite_book_nl,												    -- nl:Sjabloon:Citeer boek
        journal_nl = cite_journal_nl,											-- nl:Sjabloon:Citeer journal
        web_nl = cite_web_nl,													-- nl:Sjabloon:Citeer web

-- German
        literatur_de = cite_book_de,	       								    -- de:Vorlage:Literatur
        internetquelle_de = cite_web_de,										-- de:Vorlage:Internetquelle
        webarchiv_de = cite_web_de,

-- Finnish
        lehtiviite_fi = cite_journal_fi,										-- fi:Malline:Lehtiviite
        kirjaviite_fi = cite_book_fi,											-- fi:Malline:Kirjaviite
        verkkoviite_fi = cite_web_fi,											-- fi:Malline:Verkkoviite

-- French
        article_fr = cite_journal_fr,											-- fr:Modèle:Article
        ouvrage_fr = cite_book_fr,												-- fr:Modèle:Ouvrage
        web_fr = cite_web_fr,													-- fr:Modèle:Lien web

-- Italian
        libro_it = cite_book_it,												-- it:Template:Cita libro
        pubblicazione_it = cite_journal_it,										-- it:Template:Cita pubblicazione
        news_it = cite_news_it,												    -- en:Template:Cite news/Italian; for cita news see cite_news_es_it()
        web_it = cite_web_it,													-- it:Template:Cita web

-- Norwegian
        bok_no = cite_book_no,										  		    -- no:Mal:Kilde bok
        artikkel_no = cite_journal_no,											-- no:Mal:Kilde artikkel
        avis_no = cite_news_no,												    -- no:Mal:Kilde avis
        www_no = cite_web_no,													-- no:Mal:Kilde www

-- Polish
        cytuj_pl = citation_pl,													-- pl:Szablon:Cytuj
        pismo_pl = cite_journal_pl,											    -- pl:Szablon:Cytuj pismo
--          ["książkę_pl"] = cite_book_pl,
--          ["stronę_pl"] = cite_web_pl,
        book_pl = cite_book_pl,											        -- pl:Szablon:Cytuj książkę
        web_pl = cite_web_pl,										            -- pl:Szablon:Cytuj stronę

-- Portuguese
        livro_pt = cite_book_pt,												-- pt:Predefinição:citar livro
        jornal_pt = cite_news_pt,												-- pt:Predefinição:citar jornal
        web_pt = cite_web_pt,													-- pt:Predefinição:citar web
        link_pt = cite_web_pt,
--         [periódico_pt] = cite_journal_pt,										-- pt:Predefinição:citar periódico
        journal_pt = cite_journal_pt,

-- Spanish
        libro_es = cite_book_es,												-- es:Plantilla:Cita libro
--         "publicación_es" = cite_journal_es,						    			-- es:Plantilla:Cita publicación
        journal_es = cite_journal_es,
        news_es = cite_news_es,												    -- es:Plantilla:Cita news
        web_es = cite_web_es,													-- es:Plantilla:Cita web
        noticia_es = cite_news_es,

-- Swedish
        bokref_sv = cite_book_sv,												-- sv:Mall:Bokref
        tidskriftsref_sv = cite_journal_sv,										-- sv:Mall:Tidskriftsref
        webbref_sv = cite_web_sv,												-- sv:Mall:Webbref
        web_sv = cite_web_sv,

-- Turkish
--         ["kitap kaynağı_tr"] = cite_book_tr,									-- tr:Şablon:Kitap kaynağı
--         ["haber kaynağı_tr"] = cite_news_tr,                                    -- tr:Şablon:Haber kaynağı
--         ["akademik dergi kaynağı_tr"] = cite_journal_tr,                        -- tr:Şablon:Akademik dergi kaynağı
--         ["web kaynağı_tr"] = cite_web_tr,                                       -- tr:Şablon:Web kaynağı
        book_tr = cite_book_tr,
        news_tr = cite_news_tr,
        journal_tr = cite_journal_tr,
        web_tr = cite_web_tr,
        akademik_tr = cite_journal_tr,
        webref_tr = cite_web_tr,

-- Russian
--         ['Книга_ru'] = cite_book_ru
        book_ru = cite_book_ru,
        journal_ru = cite_journal_ru,
        media_ru = cite_media_ru,
        wayback_ru = cite_web_ru
    }

    local choice = template_name:lower() .. '_' .. lang;
--     print("Method name: ", choice)
    method = methods[choice];
    if method then
        return method(citation);
    else
        --Unknown template, try general
        print('Lua translator failed to find suitable method ('..choice..')');
    end
    return citation;
end
