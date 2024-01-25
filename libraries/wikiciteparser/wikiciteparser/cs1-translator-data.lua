--[[--------------------------< P A R A M S _ M A I N _ T >----------------------------------------------------

table of k/v_t pair tables where k/v_t in the outer table is:
	k – the Wikimedia subdomain (language code; 'en' in 'en.wikipedia.org')
	v_t – a k/v table where
		k – the non-English parameter name
		v – the directly translatable cs1|2 equivalent parameter name.  Parameter names that do not have any cs1|2
			equivalency or that are treated as special cases are also listed here for completeness and to document
			them; these are asigned nil for a value (same as not present in the table).

All key values (non-English parameter names) are normalized to lowercase.

]]

local params_main_t = {
	ca = {																		-- Catalan from :ca:Plantilla:Ref-web & Ref-publicació by [[Module:Sandbox/trappist_the_monk/wikisource_param_fetch]]
		['any'] = nil,															-- year
--		['archive-date'] = '',
		['article'] = 'title',													-- alias of |title= from Ref-publicació (cite news)
		['arxiudata'] = 'nil',													-- archive-date
		['arxiuurl'] = 'archive-url',
--		['arxiv'] = '',
--		['bibcode'] = '',
		['autorenllaç'] = 'author-link',										-- not enumerated; first author only
		['citació'] = 'quote',
		['coautors'] = nil,														-- no cs1|2 equivalent: |coauthors=
		['consulta'] = nil,														-- access-date
		['darrer'] = nil,														-- no cs1|2 equivalent: alias of |last= used only to create CITEREF id
		['data'] = nil,															-- date
		['dataaccés'] = nil,													-- access-date
--		['display-authors'] = '',
--		['doi'] = '',
		['doietiqueta'] = 'doi',												-- alias of |doi= apparently manually percent encoded (as of 2022-12-26 no examples of its use at ca.wiki)
		['edició'] = 'issue',													-- undocumented
		['editor'] = 'publisher',
		['editorial'] = 'publisher',
		['exemplar'] = 'issue',
--		['format'] = '',
--		['id'] = '',
		['idioma'] = 'language',
		['inactiu'] = nil,														-- no cs1|2 equivalent; more-or-less same as {{dead link}}
--		['issn'] = '',
--		['jstor'] = '',
		['llengua'] = 'language',
		['lloc'] = 'location',
		['mes'] = nil,															-- month
		['obra'] = 'work',
--		['oclc'] = '',
--		['pmc'] = '',
--		['pmid'] = '',
		['pàgina'] = 'page',
		['pàgines'] = 'pages',
		['publicació'] = 'work',
--		['ref'] = '',
		['revista'] = 'work',
		['títol'] = 'title',
--		['url'] = '',
		['volum'] = 'volume',
--		['year'] = '',

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['autor#'] = 'author#',
		['cognom#'] = 'last#',
		['enllaçautor#'] = 'author-link#',
--		['last#'] = '',
		['nom#'] = 'first#',
		},

	da = {																		-- Danish from :da:Modul:Citation/CS1/Whitelist
		['hentningsdato'] = nil,												-- access-date
		['hentet'] = nil,														-- access-date
		['besøgsdato'] = nil,													-- access-date
		['forfattere'] = 'authors',
		['redaktører'] = nil,													-- no cs1|2 equivalent: |editors=
		['bogtitel'] = 'book-title',
		['kartografi'] = 'cartography',
		['kapitel'] = 'chapter',
		['kapitel-format'] = 'chapter-format',
		['kapitel-url'] = 'chapter-url',
		['bidrag-url'] = 'contribution-url',
		['by'] = nil,															-- no cs1|2 equivalent: |city=
		['klasse'] = 'class',
		['medforfattere'] = nil,												-- no cs1|2 equivalent: |coauthors=
		['medforfatter'] = nil,													-- no cs1|2 equivalent: |coauthor=
		['samarbejde'] = 'collaboration',
		['konference'] = 'conference',
		['konference-format'] = 'conference-format',
		['konference-url'] = 'conference-url',
		['bidrag'] = 'contribution',
		['dødtlink'] = nil,														-- special case: |url-status=
		['vis-forfattere'] = 'display-authors',									-- accepts cs1|2 values: digits or 'etal' string so these are simple translations
		['vis-redaktører'] = 'display-editors',
		['visforfattere'] = 'display-authors',
		['visredaktører'] = 'display-editors',
		['udgave'] = 'edition',
		['encyklopædi'] = 'encyclopedia',
		['ignorer-isbn-fejl'] = 'isbn',											-- special case: |isbn=((<isbn>))
		['nummer'] = 'number',
		['sprog'] = 'language',
		['på'] = 'language',
		['kort'] = 'map',
		['minutter'] = 'minutes',
		['netværk'] = 'network',
		['originalår'] = nil,													-- orig-date; is it proper to translate this? what about non-date text?
		['andre'] = 'others',
		['side'] = 'page',
		['s'] = 'page',
		['sider'] = 'pages',
		['ss'] = 'pages',
		['tidsskrift'] = 'journal',
		['avis'] = 'newspaper',
		['magasin'] = 'magazine',
		['arbejde'] = 'work',
		['værk'] = 'work',
		['ordbog'] = 'dictionary',
		['hjemmeside'] = 'website',
		['sted'] = 'location',
		['udgivelsesdato'] = 'publication-date',
		['udgivelsessted'] = 'publication-place',
		['citat'] = 'quote',
		['målestok'] = 'scale',
		['skala'] = 'scale',
		['sektion'] = 'section',
		['årstid'] = nil,														-- not in aliases list; |season=?
		['sæson'] = 'season',
		['sektioner'] = 'sections',
		['serie'] = 'series',
		['række'] = 'series',
		['serielink'] = 'series-link',
		['serienr'] = 'series-number',
		['blad'] = 'sheet',
		['blade'] = 'sheets',
		['dato'] = nil,															-- |date=
		['abonnement'] = nil,													-- |subscription=
		['arkivdato'] = nil,													-- |archive-date
		['arkivurl'] = 'archive-url',
		['tid'] = 'time',
		['titel'] = 'title',
		['titellink'] = 'title-link',
		['udgiver'] = 'publisher',
		['utgiver'] = 'publisher',
		['forlag'] = 'publisher',
		['bureau'] = 'agency',
		['bind'] = 'volume',
		['år'] = nil,															-- |year=
		['kommentar'] = nil,													-- not in aliases list; |comment=?
		['verk'] = 'work',
		['tittel'] = 'title',
		['utgiver'] = 'publisher',
		['hämtdatum'] = nil,													-- access-date
		['dødlenke'] = nil,														-- special case: |url-status=
		['besøksdato'] = nil,													-- access-date
		['språk'] = nil,														-- special case: |language=
		['arkiv_url'] = 'archive-url',
		['utgivare'] = 'publisher',
		['datum'] = nil,														-- |date=
		['utgivelsesdato'] = 'publication-date',
		['etternavn'] = 'surname',
		['sitat'] = 'quote',
		['författare'] = 'authors',
		['titel_oversat'] = 'trans-title',
		['accesdate'] = nil,													--access-date
		['accessed'] = nil,														--access-date
		['acces date'] = nil,													--access-date
		['acessdate'] = nil,													--access-date
		['acces-date'] = nil,													--access-date
		['accessdate'] = nil,													--access-date
		['accessdato'] = nil,													--access-date
		['autor'] = 'author',
		['deadlink'] = nil,														-- special case: |url-status=
		['død-lenke'] = nil,													-- special case: |url-status=
		['langue'] = 'language',
		['lang'] = 'language',
		['publsiher'] = nil,													-- English misspelling
		['pubsliher'] = nil,													-- English misspelling
		['origdate'] = 'orig-date',												-- orig-date; is it proper to translate this? what about non-date text?
		['kvalitet'] = nil,														-- not in aliases list; |quality=?
		['utgivelsesår'] = 'publication-date',
		['utgivelsessted'] = 'publication-place',
		['udgivelsesår'] = 'publication-date',
		['artikel'] = 'article',
		['utgave'] = 'edition',
		['wikilink'] = 'author-link',
		['hentedag'] = nil,														--access-date
		['dag'] = nil,															-- |date=
		['oplag'] = 'edition',
		['andet'] = nil,														-- not in aliases list; |others=?
		['kapitelurl'] ='chapter-url',
		['separator'] = nil,													-- not in aliases list; no cs1|2 equivalent
		['seperator'] = nil,													-- not in aliases list; no cs1|2 equivalent
		['libris'] = nil,														-- not in aliases list; no cs1|2 equivalent: |books=?
		['dateformat'] = 'df',

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['fornavn#'] = 'first#',
		['forfatter#'] = 'author#',
		['efternavn#'] = 'last#',
		['forfatter#link'] = 'author#-link',
		['forfatter#maske'] = 'author#-mask',
		['redaktør#-fornavn'] = 'editor#-first',
		['redaktør#-efternavn'] = 'editor#-last',
		['redaktør#'] = 'editor#',
		['redaktør#link'] = 'editor#-link',
		['redaktør#maske'] = 'editor#-mask',
		['contributor#maske'] = 'contributor#-mask',
		['oversætter#'] = 'translator#',
		['oversætter#-fornavn'] = 'translator#-first',
		['oversætter#-efternavn'] = 'translator#-last',
		['oversætter#link'] = 'translator#-link',
		['oversætter#maske'] = 'translator#-mask',
		},

	de = {																		-- German
																				-- from {{Literatur}} (de:Vorlage:Literatur)
		['abruf'] = nil,														-- access-date
		['auflage'] = 'edition',
		['band'] = 'volume',
		['bandreihe'] = nil,													-- |series= special case; combined with reihe, nummerreihe, hrsgreihe
		['datum'] = nil,														-- date
		['fundstelle'] = nil,													-- no cs1|2 equivalent
		['hrsg'] = nil,															-- special case; meanings not the same here as in de:Vorlage:Internetquelle
		['hrsgreihe'] = nil,													-- |series= special case; combined with reihe, bandreihe, nummerreihe
		['isbnformalfalsch'] = 'isbn',											-- value is a broken but 'valid' isbn so |isbn=((<broken isbn>)); does not use isxn_make()
		['isbndefekt'] = 'isbn',												-- value is a broken but 'valid' isbn so |isbn=((<broken isbn>)); does not use isxn_make()
		['issnformalfalsch'] = 'issn',											-- value is a broken but 'valid' issn so |issn=((<broken issn>)); does not use isxn_make()
		['jahr'] = nil,															-- year; defined as 'outdated' at de.wiki
		['jahrea'] = nil,														-- no cs1|2 equivalent
		['kapitel'] = 'chapter',
		['kbytes'] = nil,														-- no cs1|2 equivalent
		['kommentar'] = nil,													-- no cs1|2 equivalent
		['lizenznummer'] = nil,													-- no cs1|2 equivalent
		['monat'] = nil,														-- month; defined as 'outdated' at de.wiki; retained here just because
		['nummer'] = 'issue',
		['nummerreihe'] = nil,													-- |series= special case; combined with reihe, bandreihe, hrsgreihe
		['online'] = 'url',
		['originaljahr'] = 'orig-date',											--orig-date; is it proper to translate this? what about non-date text?
		['originalort'] = nil,													-- no cs1|2 equivalent
		['originalsprache'] = 'language',
		['originaltitel'] = 'trans-title',
		['ort'] = 'location',
		['ortea'] = nil,														-- no cs1|2 equivalent
		['reihe'] = nil,														-- |series= special case; combined with nummerreihe, bandreihe, hrsgreihe
		['sammelwerk'] = 'periodical',
		['seiten'] = 'pages',
		['spalten'] = nil,														-- special case; no cs1|2 equivalent; see at_make()
		['tag'] = nil,															-- day; defined as 'outdated' at de.wiki; retained here just because
		['titel'] = 'title',
		['titelerg'] = 'type',
		['typ'] = nil,															-- in Vorlage:Literatur takes 'wl' as only valid value; same as cs1|2 |display-authors=0?
		['verlag'] = 'publisher',
		['verlagea'] = nil,														-- no cs1|2 equivalent
		['zitat'] = 'quote',
		['zugriff'] = nil,														-- access-date

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['autor#'] = 'author#',
		['herausgeber#'] = 'editor#',
		['übersetzer#'] = 'translator#',

																				-- from {{cite web/German}} (de:Vorlage:Internetquelle)
		['abruf-verborgen'] = nil,												-- no cs1|2 equivalent
		['archiv-bot'] = nil,													-- no cs1|2 equivalent
		['archiv-datum'] = nil,													-- archive-date
		['archiv-url'] = 'archive-url',
		['ch'] = nil,															-- no cs1|2 equivalent
		['datum-jahr'] = nil,													-- year
		['hrsg'] = nil,															-- special case; meanings not the same here as in de:Vorlage:Literatur
		['offline']= nil,														-- not quite the same as |url-status=; |offline=<anything> means |url-status=dead
		['sprache'] = 'language',
		['url'] = 'url',
		['werk'] = 'website',													-- was periodical
		['zugriff-jahr'] = nil,													-- access-date-year?
		},

	es = {																		-- these parameters taken from :es:Módulo:Citas/Configuración
		['1'] = nil,															-- no cs1|2 equivalent
		['agencia'] = 'agency',
		['ampersand'] = nil,													-- |name-list-style=amp
		['año'] = nil,															-- year
		['año-original'] = nil,													-- orig-date; is it proper to translate this? what about non-date text?
		['añoacceso'] = nil,													-- access-date
		['artículo'] = 'article',
		['autores'] = 'authors',
		['capítulo'] = 'chapter',
		['capítulo-trad'] = 'trans-chapter',
		['cita'] = 'quote',
		['ciudad'] = 'location',
		['colección'] = 'series',												-- Inexistente en la plantilla original. Añadido como sinónimo de serie.
		['conferencia'] = 'conference',
		['diccionario'] = 'dictionary',
		['edición'] = 'edition',
		['editorial'] = 'publisher',
		['en'] = 'at',
		['enciclopedia'] = 'encyclopedia',
		['enlaceeditor'] = 'editor-link',
		['enlace-pasaje'] = nil,												-- no cs1|2 equivalent: |passage-url=?
		['entrevistador'] = 'interviewer',
		['extra'] = nil,														-- no cs1|2 equivalent: |extra=? -- Inexistente en la plantilla original
		['fecha'] = nil,														-- date
		['fecha-acceso'] = nil,													-- access-date
		['fecha-doi-roto'] = nil,												-- doi-broken-date
		['fechaprofano'] = nil,													-- lay-date
		['fecha-publicación'] = nil,											-- publication-date
		['fecharesumen'] = nil,													-- lay-date
		['fecha-resumen'] = nil,												-- lay-date
		['fechaacceso'] = nil,													-- access-date
		['fechaarchivo'] = nil,													-- archive-date
		['formato'] = 'format',
		['fuenteresumen'] = 'lay-source',
		['fuenteprofano'] = 'lay-source',
		['grado'] = 'degree',
		['idioma'] = 'language',
		['isbn13'] = 'isbn',
		['ISBN13'] = 'isbn',
		['localización'] = 'location',
		['lugar'] = 'location',
		['lugar-publicación'] = 'publication-place',
		['medio'] = 'medium',
		['número'] = 'number',
		['número-autores'] = 'display-authors',									-- accepts digits only so these are simple translations
		['número-editores'] = 'display-editors',
		['obra'] = 'work',
		['otros'] = 'others',
		['página'] = 'page',
		['páginas'] = 'pages',
		['pasaje'] = nil,														-- no cs1|2 equivalent: |passage=?
		['periódico'] = 'periodical',
		['persona'] = 'authors',
		['personas'] = 'authors',
		['publicación'] = 'periodical',
		['pub-periódica'] = 'periodical',
		['puntofinal'] = 'postscript',
		['registro'] = nil,														-- special case: |url-access=registration
		['requiereregistro'] = nil,												-- special case: |url-access=registration
		['requiere-registro'] = nil,											-- special case: |url-access=registration
		['resumen'] = 'lay-url',
		['resumenprofano'] = 'lay-url',
		['revista'] = 'magazine',
		['separador'] = nil,													-- no cs1|2 equivalent
		['separador-autores'] = nil,											-- no cs1|2 equivalent
		['separador-nombres'] = nil,											-- no cs1|2 equivalent
		['serie'] = 'series',
		['sined'] = nil,														-- no cs1|2 equivalent -- Inexistente en la plantilla original
		['sinpp'] = 'no-pp',
		['sitio web'] = 'website',
		['sitioweb'] = 'website',
		['suscripción'] = nil,													-- special case: |url-access=subscription
		['temporada'] = 'season',
		['tiempo'] = 'time',
		['tipo'] = 'type',
		['título'] = 'title',													-- No pongo titre
		['títulolibro'] = 'book-title',
		['trad-título'] = 'trans-title',
		['título_trad'] = 'trans-title',
		['títulotrad'] = 'trans-title',
		['título-trad'] = 'trans-title',
		['traductor'] = 'translator',
		['traductores'] = nil,													-- no cs1|2 equivalent
		['ubicación'] = 'location',
		['ubicación-publicación'] = 'publication-place',
		['urlarchivo'] = 'archive-url',
		['url-capítulo'] = 'chapter-url',
		['urlcapítulo'] = 'chapter-url',
		['urlconferencia'] = 'conference-url',
		['urlmuerta'] = nil,													-- special case: |url-status=
		['url-pasaje'] = nil,													-- no cs1|2 equivalent
		['versión'] = 'version',
		['volumen'] = 'volume',
		['wikidata'] = nil,														-- no cs1|2 equivalent

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['máscaraautor#'] = 'author-mask#',
		['máscara-autor#'] = 'author-mask#',
		['apellido#'] = 'last#',
		['apellidos#'] = nil,													-- no cs1|2 equivalent: |authors#= or |lasts#=
		['apellido-editor#'] = 'editor-last#',
		['apellidos-editor#'] = nil,											-- no cs1|2 equivalent: |editors#=
		['autor#'] = 'author#',
		['enlaceautor#'] = 'author-link#',
		['enlace-autor#'] = 'author-link#',
		['enlace-editor#'] = 'editor-link#',
		['nombre#'] = 'first#',
		['nombre-editor#'] = 'editor-first#',
		['nombres#'] = nil,														-- no cs1|2 equivalent: |first-names=?
		},

	fi = {																		-- Finnish from fi:Malline:Verkkoviite (web), fi:Malline:Lehtiviite (journal), fi:Malline:Kirjaviite (book)
		['ajankohta'] = nil,													-- date
		['arkisto'] = 'archive-url',
		['arkistoitu'] = nil,													-- archive-date
		['ietf-kielikoodi'] = nil,												-- no cs1|2 equivalent
		['julkaisija'] = 'publisher',
		['julkaisu'] = 'work',
		['julkaisupaikka'] = 'location',
		['kappale'] = 'chapter',
		['kieli'] = 'language',
		['lainaus'] = 'quote',
		['lopetusmerkki'] = 'postscript',
		['luettu'] = nil,														-- access-date
		['luku'] = nil,															-- purportedly |number= but in use at fi.wiki it's all sorts of things; nil to get cs1|2 error message
		['nimeke'] = 'title',
		['nimike'] = 'title',
		['numero'] = 'issue',
		['osoite'] = 'url',
		['otsikko'] = 'title',
		['palsta'] = nil,														-- special case: at; |column=
		['palstat'] = nil,														-- special case: at; |columns=
		['selite'] = 'version',
		['sivu'] = 'page',
		['sivusto'] = 'work',
		['sivut'] = 'pages',
		['suomentaja'] = 'translator',
		['tekijä'] = 'author',
		['tiedostomuoto'] = 'format',
		['tunniste'] = 'id',
		['viitattu'] = nil,														-- access-date
		['vuosi'] = nil,														-- year
		['vuosikerta'] = 'volume',
		['www'] = 'url',
		['www-teksti'] = nil,													-- no cs1|2 equivalent
		},

	fr = {																		-- these taken from fr:Modèle:Article, fr:Modèle:Lien_web, fr:Ouverage
		['accès url'] = nil,													-- special case |url-access=; 'libre' free, 'inscription' subscription, 'limité' limited, 'payant' subscription (paid)
		['année'] = nil,														-- special case: year
		['année première édition'] = 'orig-date',
		['archiveurl'] = 'archive-url',
		['auteur institutionnel'] = 'author',
		['auteurs ouvrage'] = 'editor',											-- *not* 'authors'; used by fr:Chapitre
		['bnf'] = nil,															-- special case |id=; [[:fr:Bibliothèque nationale de France]] call {{BNF}}?
		['brisé le'] = nil,														-- no cs1|2 equivalent; 'broke it'? |url-status=dead / {{dead link}}
		['champ libre'] = nil,													-- no cs1|2 equivalent; free field?
		['chap'] = nil,															-- special case: chapter
		['chapitre'] = nil,														-- special case: chapter
		['citation'] = 'quote',
		['collection'] = 'series',												-- special case: book collection
		['consulte le'] = nil,													-- special case: misc dates
		['consulté le'] = nil,
		['dead-url'] = nil,														-- special case: |url-status=
		['description'] = 'type',
		['dnb'] = nil,															-- special case |id=; [[:fr:Bibliothèque nationale allemande]]; same as de DNB-IDN?
		['ean'] = nil,															-- special case |id=; [[:fr:Code-barres EAN]]
		['écouter en ligne'] = nil,												-- no cs1|2 equivalent; audio books url
		['éditeur'] = 'publisher',
		['édition'] = 'publisher',
		['et alii'] = nil,														-- accepted value 'oui'; special case |display-authors=etal
		['et al.'] = nil,														-- special case |display-authors=etal; accepted value 'oui'
		['format électronique'] = 'format',
		['format livre'] = nil,													-- no cs1|2 equivalent; physical format of the book
		['hal'] = nil,															-- special case |id=; [[:fr:HAL (archive ouverte)]]
		['id'] = 'ref',															-- not same as en.wiki |id=
		['illustrateur'] = 'others',
		['isbn erroné'] = nil,													-- special case |isbn=((<invalid isbn>))
		['jour'] = nil,															-- special case: day
		['langue'] = 'language',
		['langue originale'] = nil,												-- no cs1|2 equivalent; 'translated-from' language
		['libellé'] = nil,														-- no cs1|2 equivalenta simple display label
		['libris'] = nil,														-- special case |id=; [[:fr:LIBRIS]]
		['lien langue'] = 'language',
		['lien titre'] = 'title-link',
		['lieu'] = 'location',
		['lire en ligne'] = 'url',												-- full-text url
		['math reviews'] = 'mr',
		['mois'] = nil,															-- special case: month
		['nature article'] = 'type',
		['nature document'] = 'type',
		['nature ouvrage'] = 'type',
		['numdam'] = nil,														-- special case |id=; [[:fr:Numérisation de documents anciens mathématiques]]; apparently supported but not used
		['numéro'] = nil,														-- special case: chapter number
		['numéro article'] = nil,												-- no cs1|2 equivalent; article number
		['numéro chapitre'] = nil,												-- special case: chapter number
		['numéro dans collection'] = 'series',									-- special case: number of the book in the series
		['numéro édition'] = 'edition',
		['pages'] = nil,														-- alias of |pages totales=
		['pages totales'] = nil,												-- no cs1|2 equivalent; total number of pages in the book
		['partie'] = nil,														-- no cs1|2 equivalent; part number
		['passage'] = 'page',
		['photographe'] = 'others',												-- photographer
		['plume'] = nil,														-- no cs1|2 equivalent; 'feather'? when set to 'oui' displays icon
		['pmcid'] = 'pmc',
		['postface'] = 'contributor',											-- name of person who wrote the postscript
		['publi'] = nil,														-- no cs1|2 equivalent; reprint year(s); alias of |réimpression=
		['préface'] = 'contributor',											-- name of person who wrote the preface
		['présentation en ligne'] = nil,										-- no cs1|2 equivalent? url of presentation or review; sort of like deprecated |lay-url=?
		['périodique'] = 'periodical',
		['ref'] = nil,															-- no cs1|2 equivalent; alias of |référence simplifiée=
		['référence'] = nil,													-- no cs1|2 equivalent; link to reference in the :fr:Référence: namespace; see [[:fr:Aide:Espace référence]]
		['référence simplifiée'] = nil,											-- no cs1|2 equivalent
		['réimpression'] = nil,													-- no cs1|2 equivalent; reprint year(s)
		['résumé'] = nil,														-- no cs1|2 equivalent? url of presentation or review; sort of like deprecated |lay-url=?; alias of |présentation en ligne=
		['série'] = 'series',
		['site'] = 'website',
		['sous-titre'] = nil,													-- special case: |title=; subtitle combined with |title=
        ['sous-titre ouvrage'] = nil,   -- special case: |title=; subtitle combined with |title= (added to function title_make_fr)
		['sudoc'] = nil,														-- special case |id=; [[:fr:Système universitaire de documentation]]
		['titre'] = nil,														-- special case: |title=
		['titre chapitre'] = nil,												-- special case |chapter=
		['titre numéro'] = nil,													-- no cs1|2 equivalent; special case: issue title?
		['titre original'] = nil,												-- special case: |title=; title in original language
		['titre tome'] = 'volume',												-- special case book volume title
		['titre vo'] = nil,														-- special case: |title=; title in original language
		['titre volume'] = nil,													-- special case book volume title
		['tome'] = nil,															-- special case book volume number
		['trad'] = 'translator',
		['traducteur'] = 'translator',
		['traduction'] = 'translator',
		['traduction titre'] = 'trans-title',
		['traductrice'] = 'translator',
		['url résumé'] = nil,													-- no cs1|2 equivalent? url of presentation or review; sort of like deprecated |lay-url=?; alias of |présentation en ligne=
		['url texte'] = 'url',
		['wikisource'] = nil,													-- no cs1|2 equivalent; title of the book's wikisource page

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['auteur#'] = 'author#',
		['directeur#'] = nil,													--no cs1|2 equivalent; "If the author assumes the responsibility of director of publication, indicate yes; otherwise, do not use this parameter"
		['lien auteur#'] = 'author-link#',
		['nom#'] = 'last#',
		['postnom#'] = 'last#',													-- enumerated forms
		['prenom#'] = 'first#',
		['prénom#'] = 'first#',
		['responsabilité#'] = nil,												-- no cs1|2 equivalent; "Possible additional liability assumed by the author; if he is a publication directeur1, prefer the directeur1 parameter."

	-- cs1|2 equivalents														-- TODO: delete these
		['archive-date'] = nil,
		['archive-url'] = nil,
		['arxiv'] = nil,
		['doi'] = nil,
		['bibcode'] = nil,
		['e-issn'] = nil,
		['format'] = nil,
		['isbn'] = nil,
		['issn'] = nil,
		['jstor'] = nil,
		['lang'] = nil,
		['oclc'] = nil,
		['origyear'] = nil,
		['page'] = nil,
		['pmid'] = nil,
		['url'] = nil,
		['zbl'] = nil,
		},

	it = {																		-- this list from :it:Modulo:Citazione/Whitelist
		['DoiBroken'] = nil,													-- doi-broken-date
		['abstract'] = nil,														-- no cs1|2 equivalent
		['accesso'] = nil,														-- access-date
		['altri'] = 'others',
		['altrilink'] = nil,													-- no cs1|2 equivalent: |others-link=?
		['anno'] = nil,															-- year
		['annoaccesso'] = nil,													-- access-date
		['annodiaccesso'] = nil,												-- access-date
		['annooriginale'] = nil,												-- orig-date; is it proper to translate this? what about non-date text?
		['articolo'] = 'article',
		['autore'] = 'author',
		['bnf'] = nil,															-- no cs1|2 equivalent
		['canale'] = 'station',
		['capitolo'] = 'chapter',
		['capitolotradotto'] = 'trans-chapter',
		['cartografia'] = 'cartography',
		['cid'] = 'ref',
		['citazione'] = 'quote',
		['città'] = 'location',
		['coautore'] = nil,														-- no cs1|2 equivalent: |coauthor=
		['coautori'] = nil,														-- no cs1|2 equivalent: |coauthors=
		['codici'] = 'id',
		['cognome'] = 'author',
		['collana'] = 'periodical',
		['conferenza'] = 'conference',
		['contributo'] = 'contribution',
		['copertina'] = nil,													-- no cs1|2 equivalent: |cover=?
		['curatore'] = 'editor',
		['curatore-cognome'] = 'editor-last',
		['curatore-nome'] = 'editor-first',
		['curatori'] = nil,														-- no cs1|2 equivalent: |editors=
		['data'] = nil,															-- date
		['dataaccesso'] = nil,													-- access-date
		['dataarchivio'] = nil,													-- archive-date
		['dataarchivio2'] = nil,												-- no cs1|2 equivalent: |archive-date2=?
		['datadiaccesso'] = nil,												-- access-date
		['dataoriginale'] = nil,												-- orig-date; is it proper to translate this? what about non-date text?
		['datapubblicazione'] = nil,											-- publication-date
		['datatrasmissione'] = nil,												-- date
		['deadurl'] = nil,														-- special case: |url-status=
		['dizionario'] = 'dictionary',
		['doi_brokendate'] = nil,												-- doi-broken-date
		['doi_inactivedate'] = nil,												-- doi-broken-date
		['ed'] = 'edition',
		['editore'] = 'publisher',
		['edizione'] = 'edition',
		['enciclopedia'] = 'encyclopedia',
		['ente'] = 'publisher',
		['episodio'] = 'issue',
		['etal'] = nil,															-- special case: |display-authors=etal apparently any value (typically 's', 'sì', or 'si')
		['etalcuratori'] = nil,													-- special case: |display-editors=etal
		['evidenzia'] = nil,													-- no cs1|2 equivalent: |highlights=?
		['formato'] = 'format',
		['giornale'] = 'newspaper',
		['giorno'] = nil,														-- day
		['giornoaccesso'] = nil,												-- access-date
		['giornodiaccesso'] = nil,												-- access-date=
		['giornooriginale'] = nil,												-- orig-day=
		['i'] = nil,															-- no cs1|2 equivalent; don't know what this is for
		['ignoraisbn'] = 'isbn',												-- special case: |isbn=((value))
		['illustratore'] = nil,													-- no cs1|2 equivalent; combine with |others=? TODO
		['illustratori'] = nil,													-- no cs1|2 equivalent; combine with |others=? TODO
		['isbn13'] = 'isbn',
		['lastauthoramp'] = nil,												-- special case: |name-list-style=amp
		['laydate'] = nil,														-- lay-date
		['laysource'] = 'lay-source',
		['laysummary'] = 'lay-url',
		['layurl'] = 'lay-url',
		['lingua'] = nil,														-- special case language
		['mese'] = nil,															-- month=
		['meseaccesso'] = nil,													-- access-date=
		['mesediaccesso'] = nil,												-- access-date=
		['meseoriginale'] = nil,												-- no cs1|2 equivalent: |orig-month=
		['minuto'] = 'minutes',
		['nocat'] = 'no-tracking',
		['nopp'] = 'no-pp',
		['notracking'] = 'no-tracking',
		['numero'] = 'number',
		['opera'] = 'work',
		['ora'] = nil,															-- no cs1|2 equivalent: |hour=?
		['organizzazione'] = nil,												-- no cs1|2 equivalent: |organization=?
		['pagina'] = 'page',
		['pagine'] = 'pages',
		['periodico'] = 'periodical',
		['posizione'] = 'at',
		['posttitolo'] = nil,													-- no cs1|2 equivalent: subtitle?
		['produttore'] = 'publisher',
		['pubblicazione'] = 'periodical',
		['puntofinale'] = 'postscript',
		['richiestasottoscrizione'] = nil,										-- special case: |url-access=subscription
		['rivista'] = 'magazine',
		['romano'] = nil,														-- no cs1|2 equivalent: |roman=?
		['scala'] = 'scale',
		['secondo'] = nil,														-- no cs1|2 equivalent: |seconds=?
		['serie'] = 'series',
		['sezione'] = 'section',
		['sito'] = 'website',
		['source'] = nil,														-- no cs1|2 equivalent: source?
		['stagione'] = 'volume',
		['stile'] = nil,														-- no cs1|2 equivalent: style?
		['tempo'] = 'time',
		['tipo'] = 'type',
		['titolo'] = 'title',
		['titolooriginale'] = nil,												-- no cs1|2 equivalent: original title
		['titolotradotto'] = 'trans-title',
		['trad'] = 'translator',
		['traduttore'] = 'translator',
		['traduttori'] = nil,													-- no cs1|2 equivalent: translators?
		['trascrizione'] = 'transcript',
		['trasmissione'] = 'series',
		['url-trascrizione'] = 'transcript-url',
		['url_capitolo'] = 'chapter-url',
		['url_conferenza'] = 'conference-url',
		['urlarchivio'] = 'archive-url',
		['urlarchivio2'] = nil,													-- no cs1|2 equivalent: archive-url2?
		['urlcapitolo'] = 'chapter-url',
		['urlconferenza'] = 'conference-url',
		['urlcontributo'] = 'chapter-url',
		['urlmorto'] = nil,														-- special case: |url-status=
		['urltrascrizione'] = 'transcript-url',
		['versione'] = 'version',
		['voce'] = 'chapter',
		['vol'] = 'volume',
		['wikisource'] = nil,													-- no cs1|2 equivalent: source?
		['wkcanale'] = nil,														-- no cs1|2 equivalent: |station-link=?
		['wkcapitolo'] = nil,													-- no cs1|2 equivalent: |chapter-link=?
		['wkcuratore'] = 'editor-link',
		['wkserie'] = 'series-link',
		['wktitolo'] = 'title-link',
		['wktrasmissione'] = 'series-link',

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['autore#'] = 'author#',
		['autore-articolo#'] = nil,												-- no cs1|2 equivalent: |author-article=?
		['autore-articolo#-cognome'] = nil,										-- no cs1|2 equivalent: |author-article-surname=?
		['autore-articolo#-nome'] = nil,										-- no cs1|2 equivalent: |author-article-name=?
		['autore-articolo-cognome#'] = nil,										-- no cs1|2 equivalent: |author-article-name=?
		['autore-articolo-nome#'] = nil,										-- no cs1|2 equivalent: |author-article-name=?
		['autore-capitolo#'] = nil,												-- no cs1|2 equivalent: |author-chapter=
		['autore-capitolo#-cognome'] = nil,										-- no cs1|2 equivalent: |author-chapter-surname=?
		['autore-capitolo#-nome'] = nil,										-- no cs1|2 equivalent: |author-chapter-name=?=
		['autore-capitolo-cognome#'] = nil,										-- no cs1|2 equivalent: |author-chapter-name=?=
		['autore-capitolo-nome#'] = nil,										-- no cs1|2 equivalent: |author-chapter-name=?=
		['autore-contributo#'] = nil,											-- no cs1|2 equivalent: |author-contribution=?
		['autore-contributo#-cognome'] = nil,									-- no cs1|2 equivalent: |author-contribution-surname=?
		['autore-contributo#-nome'] = nil,										-- no cs1|2 equivalent: |author-contribution-name=?
		['autore-contributo-cognome#'] = nil,									-- no cs1|2 equivalent: |author-contribution-name=?
		['autore-contributo-nome#'] = nil,										-- no cs1|2 equivalent: |author-contribution-name=?
		['autore-sezione#'] = nil,												-- no cs1|2 equivalent: |author-section=?
		['autore-sezione#-cognome'] = nil,										-- no cs1|2 equivalent: |author-section-surname=?
		['autore-sezione#-nome'] = nil,											-- no cs1|2 equivalent: |author-section-name=?
		['autore-sezione-cognome#'] = nil,										-- no cs1|2 equivalent: |author-section-name=?
		['autore-sezione-nome#'] = nil,											-- no cs1|2 equivalent: |author-section-name=?
		['autore-voce#'] = nil,													-- no cs1|2 equivalent: |author-voice=?
		['autore-voce#-cognome'] = nil,											-- no cs1|2 equivalent: |author-voice-surname=?
		['autore-voce#-nome'] = nil,											-- no cs1|2 equivalent: |author-voice-name=?
		['autore-voce-cognome#'] = nil,											-- no cs1|2 equivalent: |author-voice-name=?
		['autore-voce-nome#'] = nil,											-- no cs1|2 equivalent: |author-voice-name=?
		['cognome#'] = 'last#',
		['curatore#'] = 'editor#',
		['curatore#-cognome'] = 'editor#-last',
		['curatore#-nome'] = 'editor#-first',
		['curatore-cognome#'] = 'editor-last#',
		['curatore-nome#'] = 'editor-first#',
		['linkautore#'] = 'author-link#',
		['nome#'] = 'first#',
		['wkautore#'] = 'author-link#',
		['wkautore-articolo#'] = nil,											-- no cs1|2 equivalent: |author-link=?
		['wkautore-capitolo#'] = nil,											-- no cs1|2 equivalent: |author-link=?
		['wkautore-contributo#'] = nil,											-- no cs1|2 equivalent: |author-link=?
		['wkautore-sezione#'] = nil,											-- no cs1|2 equivalent: |author-link=?
		['wkautore-voce#'] = nil,												-- no cs1|2 equivalent: |author-link=?
		},

	nl = {																		-- from nl:Sjabloon:Citeer web (web), nl:Sjabloon:Citeer boek (book), nl:Sjabloon:Citeer journal (journal)
		['accessdate'] = nil,													-- special case: misc dates
		['accessdaymonth'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['accessmonthday'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['accessyear'] = nil,													-- no cs1_2 equivalent; deprecated at nl.wiki
		['achternaam'] = 'last',												-- does not enumerate
		['archiefdatum'] = nil,													-- special case: misc dates
		['archiefurl'] = 'archive-url',
		['archiveurl'] = 'archive-url',
		['archivedate'] = nil,													-- special case: misc dates
		['auteur'] = 'author',													-- does not enumerate
		['auteurlink'] = 'author-link',											-- does not enumerate
		['authorlink'] = 'author-link',											-- does not enumerate
		['beozchtjaar'] = nil,													-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezochmaanddag'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezochtdag'] = nil,													-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezochtdatum'] = nil,													-- special case: misc dates
		['bezochtjaar'] = nil,													-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezochtmaanddag'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezochtmanadag'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezochtmaandag'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['bezcohtmaanddag'] = nil,												-- no cs1_2 equivalent; deprecated at nl.wiki
		['citaat'] = 'quote',
		['coauthors'] = nil,													-- no cs1|2 equivalent
		['curly'] = nil,														-- no cs1|2 equivalent; curly instead of straight quotes around a title
		['datum'] = nil,														-- special case: date
		['datumbezocht'] = nil,													-- special case: misc dates
		['datumgeraadpleegd'] = nil,											-- special case: misc dates
		['deadurl'] = nil,														-- special case: |url-access=
		['dead-url'] = nil,														-- special case: |url-access=
		['dodeurl'] = nil,														-- special case: |url-access=
		['dode-url'] = nil,														-- special case: |url-access=
		['doilabel'] = nil,														-- no cs1|2 equivalent; a label use in place of the doi when rendering the doi link
		['editie'] = 'edition',
		['formaat'] = 'format',
		['hoofdstuk'] = 'chapter',
		['hoofdstukurl'] = 'chapter-url',
		['jaar'] = nil,															-- special case: date
		['locatie'] = 'location',
		['maand'] = nil,														-- special case: date
		['medeauteurs'] = 'coauthors',											-- not supported by cs1|2; translated for the error message
		['month'] = nil,														-- special case: date
		['nur'] = nil,															-- special case: |id=
		['pagina\'s'] = 'pages',												-- special case? there is no |pagina= (page); value may include p. or pp. prefix
		['paginas'] = 'pages',
		['plaats'] = 'location',
		['publicatiedatum'] = nil,												-- special case: misc dates
		['raadpleegdatum'] = nil,												-- special case: misc dates
		['taal'] = 'language',
		['titel'] = 'title',
		['uitgever'] = 'publisher',
		['voornaam'] = 'first',													-- does not enumerate
		['werk'] = 'work',
		},

	no = {																		-- from :no:Modul:Citation/CS1/Whitelist
		['abb'] = nil,															-- |subscription=
		['abonnement'] = nil,													-- |subscription=
		['andre'] = 'others',
		['år'] = nil,															-- |year=
		['årgang'] = 'volume',
		['arkivdato'] = nil,													-- |archive-date=
		['arkiv_dato'] = nil,
		['arkiv-dato'] = nil,
		['arkivurl'] = 'archive-url',
		['arkiv_url'] = 'archive-url',
		['arkiv-url'] = 'archive-url',
		['artikkel'] = 'article',
		['avdeling'] = 'department',
		['avis'] = 'newspaper',
		['besøksdato'] = 'access-date',
		['bidrag'] = 'contribution',
		['bidragurl'] = 'contribution-url',
		['bidrag-url'] = 'contribution-url',
		['bind'] = 'volume',
		['boktittel'] = 'book-title',
		['by'] = nil,															-- no cs1|2 equivalent: |city=
		['byrå'] = 'agency',
		['nyhetsbyrå'] = 'agency',
		['dag'] = nil,															-- |day=
		['dato'] = nil,															-- |date=
		['dødlenke'] = nil,														-- |url-status=
		['død-lenke'] = nil,													-- |url-status=
		['doibrutt'] = nil,														-- doi-broken-date
		['doi_bruttdato'] = nil,												-- doi-broken-date
		['doi_inaktivdato'] = nil,												-- doi-broken-date
		['embargo'] = nil,														-- pmc-embargo-date
		['encyclopedi'] = 'encyclopedia',
		['episodelenke'] = 'episode-link',
		['forfattere'] = 'authors',
		['forfattermerke'] = nil,												-- not in aliases list
		['forfatter-merke'] = nil,												-- not in aliases list
		['forfatter_url'] = nil,												-- not in aliases list
		['forfatternavn-separator'] = nil,										-- no cs1|2 equivalent: |author-name-separator=
		['forfatterseparator'] = nil,											-- no cs1|2 equivalent: |author-separator=
		['forfatter-separator'] = nil,											-- no cs1|2 equivalent: |author-separator=
		['forlag'] = 'publisher',
		['først'] = 'air-date',
		['grad'] = 'degree',
		['hendelse'] = 'conference',
		['hendelseurl'] = 'conference-url',
		['ignorerisbnfeil'] = 'isbn',											-- special case |isbn=((isbn))
		['ignorer-isbn-feil'] = 'isbn',											-- special case |isbn=((isbn))
		['ingensporing'] = 'no-tracking',
		['ingen-sporing'] = 'no-tracking',
		['innfelt'] = 'inset',
		['institusjon'] = 'publisher',
		['isbn13'] = 'isbn',
		['kallesignal'] = nil,													-- no cs1|2 equivalent: |call-sign=
		['kapittel'] = 'chapter',
		['kapittellenke'] = nil,												-- no cs1|2 equivalent: |chapter-link=
		['kapittelurl'] = 'chapter-url',
		['kapittel-url'] = 'chapter-url',
		['kartografi'] = 'cartography',
		['konferanse'] = 'conference',
		['konferanseurl'] = 'conference-url',
		['konferanse-url'] = 'conference-url',
		['kommentar'] = nil,													-- no cs1|2 equivalent: |comment=
		['lokasjon'] = 'location',
		['magasin'] = 'magazine',
		['maldokumentasjonsdemo'] = 'no-tracking',
		['måned'] = nil,														-- |month=
		['medforfatter'] = nil,													-- no cs1|2 equivalent: |coauthor=
		['medforfattere'] = nil,												-- no cs1|2 equivalent: |coauthors=
		['media'] = 'medium',
		['medintervjuere'] = nil,												-- no cs1|2 equivalent: |cointerviewers=
		['minutter'] = 'minutes',
		['modus'] = 'mode',
		['navneseparator'] = nil,												-- no cs1|2 equivalent: |name-separator=
		['navnelisteformat'] = 'name-list-style',
		['nettside'] = 'website',
		['nettverk'] = 'network',
		['nocat'] = 'no-tracking',
		['nopp'] = 'no-pp',
		['nummer'] = 'number',
		['hefte'] = 'issue',
		['oppføring'] = 'entry',
		['opprinnelsesår'] = nil,												-- orig-date; is it proper to translate this? what about non-date text?
		['oppslagsverk'] = 'encyclopedia',
		['ordbok'] = 'dictionary',
		['overs_kapittel'] = 'trans-chapter',
		['overs-kapittel'] = 'trans-chapter',
		['overs_tittel'] = 'trans-title',
		['overs-tittel'] = 'trans-title',
		['på'] = nil,
		['periodisk'] = 'periodical',
		['personer'] = 'people',
		['program'] = nil,														-- no cs1|2 equivalent: |program=
		['publikasjon'] = 'periodical',
		['redaktør-separator'] = nil,											-- no cs1|2 equivalent: |editor-separator=; misspelled in original whitelist
		['redaktører'] = nil,													-- no cs1|2 equivalent: |editors=
		['redaktørnavn-separator'] = nil,										-- no cs1|2 equivalent: |editor-name-separator=
		['registrering'] = nil,													-- |url-access=registration; misspelled in original whitelist
		['s'] = 'page',
		['sal'] = 'docket',
		['sammendrag'] = 'lay-url',
		['sammendragdato'] = nil,												-- |lay-date=
		['sammendragkilde'] = 'lay-source',
		['sammendragurl'] = 'lay-url',
		['seksjon'] = 'section',
		['seksjonurl'] = 'section-url',
		['separator'] = nil,													-- no cs1|2 equivalent: |separator=
		['serie'] = 'series',
		['serielenke'] = 'series-link',
		['serienr'] = 'series-number',
		['serienummer'] = 'series-number',
		['serier'] = 'series',
		['serie-separator'] = nil,												-- no cs1|2 equivalent: |series-separator=
		['sesong'] = 'season',
		['side'] = 'page',
		['sideantall'] = nil,													-- not in aliases list: |number-of-pages=?;  -- kept for backwards compability, not part of CS1
		['sider'] = 'pages',
		['sisteforfatteramp'] = nil,											-- |name-list-style=amp
		['sitat'] = 'quote',
		['sitering'] = 'quote',
		['skala'] = 'scale',
		['skole'] = 'publisher',
		['språk'] = nil,														-- special case: |language=
		['sprefiks'] = nil,														-- no cs1|2 equivalent: |P-prefix=
		['ss'] = 'pages',
		['SSPrefiks'] = nil,													-- no cs1|2 equivalent: |PP-prefix=
		['sted'] = 'location',
		['tidspunkt'] = 'time',
		['tidstekst'] = 'time-caption',
		['tittel'] = 'title',
		['tittellenke'] = 'title-link',
		['transkripsjon'] = 'transcript',
		['transkripsjonsurl'] = 'transcript-url',
		['transkripsjon-url'] = 'transcript-url',
		['url-tilgang'] = 'url-access',
		['utgave'] = 'edition',
		['utgivelsesår'] = nil,													-- |year=
		['utgivelsesdato'] = nil,												-- |publication-date=
		['utgivelses-dato'] = nil,												-- |publication-date=
		['utgivelsessted'] = 'publication-place',
		['utgivelses-sted'] = 'publication-place',
		['utgiver'] = 'publisher',
		['utgiverid'] = 'id',
		['ved'] = 'at',
		['verk'] = 'work',
		['versjon'] = 'version',
		['visforfattere'] = 'display-authors',									-- accepts cs1|2 values: digits or 'etal' string so these are simple translations
		['vis-forfattere'] = 'display-authors',
		['visredaktører'] = 'display-editors',
		['vis-redaktører'] = 'display-editors',
		['volum'] = 'volume',

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['etternavn#'] = 'last#',
		['forfatter#'] = 'author#',
		['forfattere#'] = nil,													-- no cs1|2 equivalent: |authors#=
		['forfatter-etternavn#'] = 'author-surname#',
		['forfatter#-etternavn'] = 'author#-surname',
		['forfatter-fornavn#'] = 'author-given#',
		['forfatter#-fornavn'] = 'author#-given',
		['forfatterlenke#'] = 'author-link#',
		['forfatter-lenke#'] = 'author-link#',
		['forfatter#lenke'] = 'author#-link',
		['forfatter#-lenke'] = 'author#-link',
		['forfatter-merke#'] = nil,												-- not in aliases list
		['forfatter#merke'] = nil,												-- not in aliases list
		['forfatter#-merke'] = nil,												-- not in aliases list
		['forfattermerke#'] = nil,												-- not in aliases list
		['fornavn#'] = 'first#',
		['oversetter#'] = 'translator#',
		['oversetter-fornavn#'] = 'translator-first#',
		['oversetter#-fornavn'] = 'translator#-first',
		['oversetter-etternavn#'] = 'translator-last#',
		['oversetter#-etternavn'] = 'translator#-last',
		['oversetter-lenke#'] = 'translator-link#',
		['oversetter#-lenke'] = 'translator#-link',
		['oversetter-maske#'] = 'translator-mask#',
		['oversetter#-maske'] = 'translator#-mask',
		['redaktør#'] = 'editor#',
		['redaktører#'] = nil,													-- no cs1|2 equivalent: |editors#=
		['redaktør-etternavn#'] = 'editor-last#',
		['redaktør#-etternavn'] = 'editor#-last',
		['redaktøretternavn#'] = 'editor-last#',
		['redaktør-fornavn#'] = 'editor-first#',
		['redaktør#fornavn'] = 'editor#-first',
		['redaktør#-fornavn'] = 'editor#-first',
		['redaktørfornavn#'] = 'editor-first#',
		['redaktør-lenke#'] = 'editor-link#',
		['redaktør#lenke'] = 'editor#-link',
		['redaktør#-lenke'] = 'editor#-link',
		['redaktørlenke#'] = 'editor-link#',
		['redaktør-merke#'] = nil,												-- not in aliases list
		['redaktør#merke'] = nil,												-- not in aliases list
		['redaktør#-merke'] = nil,												-- not in aliases list
		['redaktørmerke#'] = nil,												-- not in aliases list
		},

	pl = {																		-- these parameters from pl:Szablon:Cytuj stronę (cite web/Polish)
		['archiwum'] = 'archive-url',
		['cytat'] = 'quote',
		['data'] = nil,															-- special case: |date=
		['data dostępu'] = nil,													-- special case: misc dates: |access-date=
--		['id'] = '',
		['miesiąc'] = nil,														-- special case: |month=
		['odn'] = nil,															-- special case: |ref=; |odn=tak is more or less like |ref=harv
		['opublikowany'] = 'website',											-- special case: |opublikowany= more-or-less same as |praca=?
		['oznaczenie'] = 'number',
		['praca'] = 'work',														-- special case: |praca= more-or-less same as |opublikowany=?
		['rok'] = nil,															-- special case: |year=
		['strony'] = 'page',
		['tytuł'] = 'title',
--		['url'] = '',
		['zaprezentowany'] = 'publisher',
		['zarchiwizowano'] = nil,												-- special case: misc dates: |archive-date=

																				-- these parameters from pl:Szablon:Cytuj (sort of a {{citation}} equivalent?)
		['czasopismo'] = 'journal',
		['dostę'] = nil,														-- special case: |url-access= values: 'o' → free, 'z' → subscription, 'r' → registration, 'c' → limited
		['inni'] = 'others',
--		['isbn'] = '',
--		['issn'] = '',
		['kropka'] = nil,														-- no cs1|2 equivalent; additional text
		['miejsce'] = 'location',
		['numer'] = 'number',
		['odpowiedzialność'] = nil,												-- no cs1|2 equivalent; person legally responsible for publication?
		['opis'] = nil,															-- no cs1|2 equivalent; 'description'
		['patent'] = nil,														-- no cs1|2 equivalent; 'patent number'
		['rozdział'] = 'chapter',
		['s'] = 'page',
		['typ nośnika'] = 'format',
		['wolumin'] = 'volume',
		['wydanie'] = 'edition',
		['wydawca'] = 'publisher',

																				-- these parameters from :pl:Szablon:Cytuj książkę ({{cite book/Polish}})
		['adres rozdziału'] = 'chapter-url',
		['część'] = nil,														-- no cs1|2 equivalent; 'part'
		['kolumny'] = nil,														-- no cs1|2 equivalent; 'columns'
		['seria'] = 'series',
		['tom'] = nil,															-- special case |volume=; volume number
		['tytuł części'] = nil,													-- no cs1|2 equivalent; 'part title'
		['tytuł tomu'] = nil,													-- special case |volume=; volume title

																				-- these parameters from :pl:Szablon:Cytuj pismo ({{cite journal/Polish}})
		['adres czasopisma'] = nil,												-- no cs1|2 equivalent; 'journal url'

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['autor#'] = 'author#',
		['autor link#'] = 'author-link#',
		['autor r#'] = 'contributor#',											-- 'chapter' author; 'contributor' here to distingush from |autor= (|author=); from {{Cytuj książkę}}
		['autor r# link'] = 'contributor-link#',								-- 'chapter' author link
		['imię#'] = 'first#',
		['imię# r'] = 'contributor-first#',										-- 'chapter' author; 'contributor' here to distingush from |imię= (|first=); from {{Cytuj książkę}}
		['język#'] = 'language#',												-- special case |language=
		['nazwisko#'] = 'last#',
		['nazwisko# r'] = 'contributor-last#',									-- 'chapter' author; 'contributor' here to distingush from |nazwisko= (|last=); from {{Cytuj książkę}}
		['redaktor#'] = 'editor#',												-- enumeration supported here even though not documented at pl.wiki
		},

	pt = {
		['accessodata'] = nil,													-- special case misc dates: 'AccessDate'
		['acessadoem'] = nil,													-- special case misc dates: 'AccessDate'
		['acessdate'] = nil,													-- special case misc dates: 'AccessDate'
		['acesso'] = nil,														-- special case misc dates: 'AccessDate'
		['acesso-data'] = nil,													-- special case misc dates: 'AccessDate'
		['acesso-url'] = nil,													-- special case 'UrlAccess'
		['acessodata'] = nil,													-- special case misc dates: 'AccessDate'
		['acessomesdia'] = nil,													-- special case: date: 'Month',
		['acessourl'] = nil,													-- special case 'UrlAccess'
		['agencia'] = 'agency',
		['agência'] = 'agency',
--		['air-date'] = 'AirDate',
--		['airdate'] = 'AirDate',
--		['airdate'] = 'Date',
		['ano'] = nil,															-- special case: date: 'Year',
		['anooriginal'] = nil,													-- special case misc dates: 'OrigYear',
		['argumentistas'] = 'people',											-- 'screenwriters'
		['arquivo-data'] = nil,													-- special case misc dates: 'ArchiveDate',
		['arquivo-url'] = 'archive-url',
		['arquivodata'] = nil,													-- special case misc dates: 'ArchiveDate',
		['arquivoformato'] = 'archive-format',
		['arquivourl'] = 'archive-url',
		['artigo'] = 'article',
		['autores'] = 'authors',
		['año'] = nil,															-- special case: date: 'Year',
--		['book-title'] = 'BookTitle',
		['booktitle'] = 'book-title',
		['call-sign'] = nil,													-- no cs1|2 equivalent
		['callsign'] = nil,														-- no cs1|2 equivalent
		['capitulo'] = 'chapter',
		['capitulo-translit'] = 'script-chapter',
		['capítulo'] = 'chapter',
		['capítulo-trad'] = 'trans-chapter',
		['capítulo-url'] = 'chapter-url',
		['capítulourl'] = 'chapter-url',
		['chapterurl'] = 'chapter-url',
		['cidade'] = nil,														-- no cs1|2 equivalent
		['cita'] = 'quote',
		['citacao'] = 'quote',
		['citar'] = 'quote',
		['citação'] = 'quote',
		['city'] = nil,															-- no cs1|2 equivalent
		['class'] = 'class',
		['classe'] = 'class',
		['coauthor'] = nil,														-- no cs1|2 equivalent
		['coauthors'] = nil,													-- no cs1|2 equivalent
		['coautor'] = nil,														-- no cs1|2 equivalent
		['coautores'] = nil,													-- no cs1|2 equivalent
		['codling'] = nil,														-- special case: 'Language',
		['colaboração'] = 'collaboration',
		['colecao'] = 'series',
		['coleção'] = 'series',
		['coleção'] = 'series',
		['conference'] = 'conference',
		['conference-format'] = 'conference-format',
		['conference-url'] = 'conference-url',
		['conferenceurl'] = 'conference-url',
		['conferencia'] = 'conference',
		['conferencia-url'] = 'conference-url',
		['consulta'] = nil,														-- special case misc dates: 'AccessDate',
		['contribuicao'] = 'contribution',
		['contribuidor'] = 'contributor',
		['contribuição'] = 'contribution',
		['contributionurl'] = 'ChapterURL',
--		['credits'] = 'Authors',
		['créditos'] = 'credits',
		['data'] = nil,															-- special case: date: 'Date',
		['data-publicacao'] = nil,												-- special case: misc dates: 'publication-date',
		['data-publicação'] = nil,												-- special case: misc dates: 'publication-date',
		['data-resumo'] = nil,													-- no cs1|2 equivalent 'LayDate',
		['data2'] = nil,														-- no cs1|2 equivalent
		['dataacesso'] = nil,													-- special case misc dates: 'AccessDate',
		['dataemissao'] = nil,													-- special case: 'Date': 'issuance date'
		['datali'] = nil,														-- special case: 'DeadURL',
		['dead-url'] = nil,														-- special case: 'DeadURL',
		['deadurl'] = nil,														-- special case: 'DeadURL',
		['departamento'] = 'department',
		['diccionario'] = 'dictionary',
		['dicionario'] = 'dictionary',
		['dicionário'] = 'dictionary',
		['displayauthors'] = 'display-authors',
		['displayeditors'] = 'display-editors',
		['distributor'] = 'publisher',
		['docket'] = 'Docket',
		['doi-broken'] = 'DoiBroken',
		['doi-inactive-date'] = 'DoiBroken',
		['doi-inactivedate'] = 'DoiBroken',
		['doi-incorrecto'] = 'DoiBroken',
		['ed'] = 'edition',
		['edicao'] = 'edition',
		['edición'] = 'edition',
		['editora'] = 'publisher',
		['editora'] = 'publisher',
		['editores'] = 'editors',
		['editorial'] = 'publisher',
		['editors'] = 'editors',
		['edição'] = 'edition',
		['edição'] = 'edition',
		['em'] = 'at',
		['embargo'] = 'pmc-embargo-date',
		['en'] = 'at',
		['enciclopedia'] = 'encyclopedia',
		['enciclopédia'] = 'encyclopedia',
		['entrevistadores'] = nil,												-- no cs1|2 equivalent: 'Interviewers',
--		['entrevistadores'] = nil,												-- no cs1|2 equivalent: 'Others',
--		['episode-link'] = 'TitleLink',
		['episodelink'] = 'episode-link',
		['episódiolink'] = 'episode-link',
		['escala'] = 'scale',
		['estacao'] = 'station',
		['estação'] = 'station',
		['event'] = 'event',
		['event-format'] = 'conference-format',
		['event-url'] = 'conference-url',
		['eventurl'] = 'conference-url',
		['expediente'] = 'docket',
		['fecha'] = nil,														-- special case: 'Date',
		['fecha-publicación'] = nil,											-- special case: misc dates: 'publication-date',
		['fechaacceso'] = nil,													-- special case: misc dates: 'AccessDate',
		['fonte-resumo'] = nil,													-- no cs1|2 equivalent 'LaySource',
		['formato'] = 'Format',
		['formato-arquivo'] = 'archive-format',
		['formato-autor'] = nil,												-- special case: 'name-list-style'
		['formato-capitulo'] = 'chapter-format',
		['formato-conferencia'] = 'conference-format',
		['formato-editor'] = nil,												-- special case: 'name-list-style'
		['formato-lista-nomes'] = nil,											-- special case: 'name-list-style'
		['formato-resumo'] = nil,												-- no cs1|2 equivalent 'LayFormat',
		['grupo-noticias'] = 'newsgroup',
		['id-mensagem'] = 'message-id',
		['idioma'] = nil,														-- special case: 'Language',
		['idioma2'] = nil,														-- special case: 'Language',
		['idioma3'] = nil,														-- special case: 'Language',
		['idioma4'] = nil,														-- special case: 'Language',
		['ignore-isbn'] = nil,													-- special case: |isbn=((<isbn>))
		['ignore-isbn-error'] = nil,											-- special case: |isbn=((<isbn>))
		['ignoreisbnerror'] = nil,												-- special case: |isbn=((<isbn>))
		['in'] = nil,															-- special case: 'Language',
		['indicativo'] = nil,													-- no cs1|2 equivalent: 'Callsign',
		['inset'] = 'inset',
		['instituicao'] = 'institution',
		['instituição'] = 'institution',
		['interviewers'] = nil,													-- no cs1|2 equivalent: 'Interviewers',
		['jornal'] = 'journal',
		['last-author-amp'] = nil,												-- special case: 'name-list-style'
		['lastauthoramp'] = nil,												-- special case: 'name-list-style'
		['lay-summary'] = nil,													-- no cs1|2 equivalent 'LayURL',
		['laydate'] = nil,														-- no cs1|2 equivalent 'LayDate',
		['laysource'] = nil,													-- no cs1|2 equivalent 'LaySource',
		['laysummary'] = nil,													-- no cs1|2 equivalent 'LayURL',
		['layurl'] = nil,														-- no cs1|2 equivalent 'LayURL',
		['lengenda'] = 'time-caption',
		['li'] = nil,															-- special case: 'DeadURL',
		['ligação inactiva'] = nil,												-- special case: 'DeadURL',
		['ligação inativa'] = nil,												-- special case: 'DeadURL',
		['ling'] = nil,															-- special case: 'Language',
		['lingua'] = nil,														-- special case: 'Language',
		['lingua2'] = nil,														-- special case: 'Language',
		['lingua3'] = nil,														-- special case: 'Language',
		['lingua4'] = nil,														-- special case: 'Language',
		['local publicação'] = 'publication-place',
		['local'] = 'location',
		['local-publicacao'] = 'publication-place',
		['local-publicação'] = 'publication-place',
		['localização'] = 'location',
		['lugar'] = 'location',
		['lugar-publicación'] = 'publication-place',
		['língua'] = nil,														-- special case: 'Language',
		['língua2'] = nil,														-- special case: 'Language',
		['língua3'] = nil,														-- special case: 'Language',
		['língua4'] = nil,														-- special case: 'Language',
		['medio'] = 'medium',
		['mensagem-id'] = 'message-id',
		['mes'] = nil,															-- special case: date: 'Month'
--		['message-id'] = 'MessageID',
		['minuto'] = 'minutes',
		['minutos'] = 'minutes',
		['modo'] = 'mode',
		['month'] = nil,														-- special case: date: 'Month'
		['mês'] = nil,															-- special case: date: 'Month'
		['name-list-format'] = nil,												-- special case: 'name-list-style'
--		['network'] = 'Network',
--		['newsgroup'] = 'PublisherName',
		['nocat'] = 'no-tracking',
		['nopp'] = 'NoPP',
		['notas'] = nil,														-- no cs1|2 equivalent; 'notes'; certainly not 'Others',
		['notracking'] = 'no-tracking',
		['numero'] = 'number',
		['numero-autores'] = 'display-authors',
		['numero-editores'] = 'display-editors',
		['numero-serie'] = 'series-number',
		['número'] = 'number',
		['obra'] = 'work',
		['otros'] = 'others',
		['outros'] = 'others',
		['pagina'] = 'page',
		['paginas'] = 'pages',
		['periodico'] = 'periodical',
		['periódico'] = 'periodical',
		['persona'] = 'people',
		['personas'] = 'people',
		['pessoas'] = 'people',
		['pontofinal'] = 'postScript',
		['produtora'] = 'publisher',
		['program'] = nil,														-- no cs1|2 equivalent: 'Program',
		['programa'] = nil,														-- no cs1|2 equivalent: 'Program',
		['publicacao'] = 'periodical',
		['publicación'] = 'periodical',
		['publicado por'] = 'publisher',
		['publicado'] = 'publisher',
		['publicadopor'] = 'publisher',
		['publicationdate'] = nil,												-- special case: misc dates: 'publication-date',
		['publicationplace'] = 'publication-place',
		['publicação'] = 'periodical',
		['página'] = 'page',
		['página'] = 'page',
		['páginas'] = 'pages',
		['rede'] = 'network',
		['registo'] = nil,														-- special case: |url-access= 'RegistrationRequired',
		['registration'] = nil,													-- special case: |url-access= 'RegistrationRequired',
		['registro'] = nil,														-- special case: |url-access= 'RegistrationRequired',
		['requadro'] = 'inset',
		['resumo'] = nil,														-- no cs1|2 equivalent 'LayURL',
		['resumo-data'] = nil,													-- no cs1|2 equivalent 'LayDate',
		['resumo-fonte'] = nil,													-- no cs1|2 equivalent 'LaySource',
		['resumo-formato'] = nil,												-- no cs1|2 equivalent 'LayFormat',
		['resumo-url'] = nil,													-- no cs1|2 equivalent 'LayURL',
		['revista'] = 'magazine',
		['season'] = 'season',
		['secao'] = 'section',
		['seccao'] = 'section',
		['secoes'] = nil,														-- no cs1|2 equivalent: 'Sections',
		['sections'] = nil,														-- no cs1|2 equivalent: 'Sections',
		['sectionurl'] = 'section-url',
		['separador-series'] = nil,												-- no cs1|2 equivalent: 'SeriesSeparator',
		['serie'] = 'series',
--		['series-link'] = 'SeriesLink',
--		['series-no'] = 'SeriesNumber',
--		['series-number'] = 'SeriesNumber',
		['series-separator'] = nil,												-- no cs1|2 equivalent: 'SeriesSeparator',
		['serieslink'] = 'series-link',
		['seriesno'] = 'series-no',
		['seriesnumber'] = 'series-number',
		['seção'] = 'section',
		['seções'] = nil,														-- no cs1|2 equivalent: 'Sections'
		['site'] = 'website',
--		['station'] = 'Station',
		['subscricao'] = nil,													-- special case: |url-access= 'SubscriptionRequired',
		['subscription'] = nil,													-- special case: |url-access= 'SubscriptionRequired',
		['subscrição'] = nil,													-- special case: |url-access= 'SubscriptionRequired',
		['subtitulo'] = nil,													-- special case: |title= with subtitle
		['subtítulo'] = nil,													-- special case: |title= with subtitle
		['suscripción'] = nil,													-- special case: |url-access= 'SubscriptionRequired',
		['série'] = 'series',
		['sérielink'] = 'series-link',
		['séries'] = 'series',
		['template doc demo'] = 'no-tracking',
		['tempo'] = 'time',
		['temporada'] = 'season',
		['tiempo'] = 'time',
		['timecaption'] = 'time-caption',
		['tipo'] = 'type',
		['titlelink'] = 'title-link',
		['titlo'] = nil,														-- special case: 'Title'
		['titulo'] = nil,														-- special case: 'Title'
		['titulo-translit'] = 'script-title',
		['titulolink'] = 'title-link',
		['titulolivro'] = 'book-title',
		['titulotrad'] = 'trans-title',
		['total-paginas'] = nil,												-- no cs1|2 equivalent: 'TotalPages',
		['total-páginas'] = nil,												-- no cs1|2 equivalent: 'TotalPages',
		['trabalho'] = 'work',
		['trad-capitulo'] = 'trans-chapter',
		['trans_chapter'] = 'trans-chapter',
		['trans_title'] = 'trans-title',
		['transcricao'] = 'transcript',
		['transcricao-formato'] = 'transcript-format',
		['transcricaourl'] = 'transcript-url',
--		['transcript'] = 'Transcript',
--		['transcript-format'] = 'TranscriptFormat',
--		['transcript-url'] = 'TranscriptURL',
		['transcripturl'] = 'transcript-url',
		['transcrição'] = 'transcript',
		['transcrição-formato'] = 'transcript-format',
		['transcriçãourl'] = 'transcript-url',
		['transmissão'] = nil,													-- special case: misc dates: 'AirDate',
		['título'] = nil,														-- special case: 'Title'
		['título'] = nil,														-- special case: 'Title'
		['título-livro'] = 'book-title',
		['título-trad'] = 'trans-title',
		['título-translit'] = 'script-title',
		['títulolink'] = 'title-link',
		['títulolivro'] = 'book-title',
		['títulotrad'] = 'trans-title',
		['ultimoamp'] = nil,													-- special case: 'name-list-style'
		['universidade'] = 'publisher',
		['url-capítulo'] = 'chapter-url',
		['url-resumo'] = nil,													-- no cs1|2 equivalent 'LayURL',
		['urlarchivo'] = 'archive-url',
		['urlarquivo'] = 'archive-url',
		['urlcapitulo'] = 'chapter-url',
		['urlcapítulo'] = 'chapter-url',
		['urlconferencia'] = 'conference-url',
		['urlmorta'] = nil,														-- special case: 'DeadURL',
		['urltranscricao'] = 'transcript-url',
		['vautores'] = 'vauthors',
		['veditores'] = 'veditors',
		['versão'] = 'series',
		['volumen'] = 'volume',
		['wayb'] = nil,															-- no cs1|2 equivalent: 'Wayb',


	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['apelido#'] = 'last#',
		['apellido#'] = 'last#',
		['apellidos#'] = nil,													-- no cs1|2 equivalent: plural |lasts#=
		['author#mask'] = 'author#-mask',
		['authormask#'] = 'author-mask#',
		['autor#'] = 'author#',
		['autor-mascara#'] = 'author-mask#',
		['autorlink#'] = 'author-link#',
		['cognome#'] = 'last#',
		['contribuidor#-primeiro'] = 'contributor#-first',
		['contribuidor#-ultimo'] = 'contributor#-last',
		['contribuidor-link#'] = 'contributor-link#',
		['contribuidor-mascara#'] = 'contributor-mask#',
		['contribuidor-primeiro#'] = 'contributor-first#',
		['contributor-ultimo#'] = 'contributor-last#',
		['editor#link'] = 'editor#-link',
		['editor#mask'] = 'editor#-mask',
		['editor-mascara#'] = 'editor#-mask',
		['editor-nome#'] = 'editor-first#',
		['editor-sobrenome#'] = 'editor-last#',
		['editorlink#'] = 'editor-link#',
		['editormask#'] = 'editor-mask#',
		['entrevistado#'] = 'interviewer#',
		['entrevistadolink#'] = 'interviewer-link#',
		['nombre#'] = 'first#',
		['nome#'] = 'first#',
		['prenome#'] = 'first#',
		['primeiro#'] = 'first#',
		['sobrenome#'] = 'last#',
		['subjectlink#'] = 'subject-link#',
		['sujeito#'] = 'subject-last#',
		['sujeitolink#'] = 'subject-link#',
		['tradutor#'] = 'translator#',
		['tradutor#-link'] = 'translator#-link',
		['tradutor#-mascara'] = 'translator#-mask',
		['tradutor#-primeiro'] = 'translator#-first',
		['tradutor#-ultimo'] = 'translator#-last',
		['tradutor-link#'] = 'translator-link#',
		['tradutor-mascara#'] = 'translator-mask#',
		['tradutor-primeiro#'] = 'translator-first#',
		['tradutor-ultimo#'] = 'translator-last#',
		['ultimo#'] = 'last#',
		['último#'] = 'last#',
		},

  	ru = {																		-- these parameters from ru:Шаблон:Книга (Template:Книга) (cite book)
		['автор'] = 'author',
		['часть'] = 'chapter',
		['ссылка'] = 'url',
		['ссылка часть'] = 'chapter-url',
		['название'] = 'title',
		['заглавие'] = 'title',
		['викитека'] = nil,														-- no cs1|2 equivalent: wikisource; |title-link=?
		['викисклад'] = nil,													-- no cs1|2 equivalent: commons; |title-link=?
		['оригинал'] = 'orig-date',
		['ответственный'] = 'agency',											-- google translate says 'responsible'; this is a cite book template, agency does not really belong here
		['издание'] = 'edition',
		['тираж'] = nil,														-- part of edition (circulation?)
		['город'] = 'location',
		['место'] = 'location',
		['год'] = 'date',
		['издательство'] = 'publisher',
		['страницы как есть'] = 'at',											-- 'pages as they are'?
		['том'] = 'volume',
		['том как есть'] = 'volume',											-- 'the way it is'?
		['выпуск'] = 'issue',													-- in {{Книга}} but not in ru:Шаблон:Книга
		['страницы'] = 'pages',
		['страниц'] = 'pages',
		['страница'] = 'page',													-- in {{Книга}} but not in ru:Шаблон:Книга
		['серия'] = 'series',
		['язык'] = 'language',
		['nodot'] = nil,														-- no cs1|2 equivalent; suppresses the dot when |title=<title> ends with puctuation
		['nodot2'] = nil,														-- no cs1|2 equivalent
		['столбцы'] = nil,														-- columns; |at=col. <column>?

		 																		-- these taken from :ru:Шаблон:Статья (Template:Книга) (cite journal)
		['автор издания'] = 'authors',
		['тип'] = 'type',
		['месяц'] = nil,														-- month
		['число'] = nil,														-- day
		['выпуск'] = 'issue',
		['номер'] = nil,														-- edition number?
		['archiveurl'] = 'archive-url',
		['archivedate'] = 'archive-date',
		},

	sv = {																		-- from sv:Mall:Webbref (web), sv:Mall:Bokref (book), sv:Mall:Tidskriftsref (journal)
		['arkivurl'] = 'archive-url',
		['citat'] = 'quote',
		['datumformat'] = nil,													-- special case: |df=?
		['doi_brokendate'] = 'doi-broken-date',
		['författarsep'] = nil,													-- special case: |name-list-style=amp
		['hämtår'] = nil,														-- no cs1|2 equivalent: |access-year=
		['hämtmånad'] = nil,													-- no cs1|2 equivalent: |access-month=
		['kapitel'] = 'chapter',
		['kapitelurl'] = 'chapter-url',
		['libris'] = nil,														-- special case: |id=
		['medförfattare'] = nil,												-- no cs1|2 equivalent: |coauthor=; at sv.wiki this is |author2=; do that?
		['nummer'] = 'number',
		['övriga'] = 'others',
		['redaktör'] = 'editor',												-- does not enumerate
		['rubrik'] = 'title',													-- journal article title
		['sammanfattning'] = nil,												-- no cs1|2 equivalent: |lay-summary=
		['sammanfattningsdatum'] = nil,											-- no cs1|2 equivalent: |lay-date=
		['separator'] = nil,													-- no cs1|2 equivalent
		['seperator'] = nil,													-- no cs1|2 equivalent
		['serie'] = 'series',
		['sid'] = 'pages',														-- an abbreviation that means page or pages?
		['sida'] = 'page',
		['sidor'] = 'pages',
		['språk'] = nil,														-- special case: |language=
		['tidskrift'] = 'journal',
		['titel'] = 'title',
		['upplaga'] = 'edition',
		['utgivare'] = 'publisher',
		['utgivningsort'] = 'location',
		['utgåva'] = 'edition',
		['verk'] = 'work',
		['volym'] = 'volume',
		['website'] = 'url',													-- different from en.wiki

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['efternamn#'] = 'last#',
		['författare#'] = 'author#',
		['författarlänk#'] = 'author-link#',
		['förnamn#'] = 'first#',
		['redaktörlänk#'] = 'editor-link#',
		['redaktör#-efternamn'] = 'editor#-last',
		['redaktör#-förnamn'] = 'editor#-first',
		},

	tr = {																		-- Turkish from :tr:Modül:Kaynak/KB1/Beyazliste
		['ansiklopedi'] = 'encyclopedia',
		['aboneli'] = nil,														-- whitelisted but not listed as an alias of anything; subscription
		['ağ'] = 'network',
		['alıntı'] = 'quote',
		['ajans'] = 'agency',
		['arşiv-tarihi'] = nil,													-- archive-date=
		['arşivtarihi'] = nil,													-- archive-date=
		['arşiv-url'] = 'archive-url',
		['arşivurl'] = 'archive-url',
		['arşiv-biçimi'] = 'archive-format',
		['arşivengelli'] = nil,													-- no cs1|2 equivalent: archive disabled
		['arşivengeli'] = nil,													-- no cs1|2 equivalent: archive blocked
		['ay'] = nil,															-- month
		['basım'] = 'edition',
		['baskı'] = 'edition',
		['başlık'] = 'title',
		['başlıkyok'] = nil,													-- no cs1|2 equivalent: no title
		['başlıkbağı'] = 'title-link',
		['başlık-bağı'] = 'title-link',
		['başlık-bağlantısı'] = 'title-link',
		['başlıklink'] = 'title-link',
		['başlıknotu'] = 'department',
		['başlıktürü'] = 'type',
		['biçim'] = 'format',
		['bölüm'] = 'episode',													-- sadece bölüm kaynağında kullanılır
		['bölümbağı'] = nil,													-- not in aliases list -- sadece bölüm kaynağında kullanılır
		['bölüm-bağı'] = nil,													-- not in aliases list -- sadece bölüm kaynağında kullanılır
		['bölüm-biçimi'] = 'section-format',
		['bölüm-url'] = 'section-url',
		['bölümurl'] = 'section-url',
		['bölüm-url-erişimi'] = 'section-url-access',
		['bölümurlerişimi'] = 'section-url-access',
		['cilt'] = 'volume',
		['cimerbaşvuruno'] = nil,												-- no cs1|2 equivalent; CİMER başvuru numarası (CIMER application number)
		['çağrıişareti'] = 'publisher',											-- call sign
		['çalışma'] = 'work',
		['çeviribaşlık'] = 'trans-title',
		['çeviri_başlık'] = 'trans-title',
		['çeviri-başlık'] = 'trans-title',
		['çeviri_bölüm'] = 'trans-section',
		['çeviri_bölümü'] = 'trans-section',
		['çeviribölüm'] = 'trans-section',
		['çeviribölümü'] = 'trans-section',										-- translation department?
		['dakika'] = 'minutes',
		['departman'] = nil,													-- not in aliases list; |department=?
		['dergi'] = 'magazine',
		['diğertarih'] = nil,													-- lay-date
		['diğer-kaynak'] = 'lay-source',
		['diğerkaynak'] = 'lay-source',
		['diğer-biçim'] = 'lay-format',
		['diğer-url'] = 'lay-url',
		['diğerurl'] = 'lay-url',
		['diğerözet'] = 'lay-url',
		['diğerleri'] = 'others',
		['dil'] = 'language',
		['doi-hatalı-tarih'] = 'doi-broken-date',
		['doi-kırık-tarih'] = 'doi-broken-date',
		['doikırıktarihi'] = nil,												-- not in aliases list; |doi-broken-date=?
		['doibozuk'] = 'doi-broken-date',
		['doi_bozuktarihi'] = 'doi-broken-date',
		['döküm'] = nil,														-- not in aliases list; |cast=?
		['döküm-url'] = nil,													-- not in aliases list; |cast-url=?
		['ebilgiedinmeno'] = nil,												-- no cs1|2 equivalent: e-Information Number?
		['edilekçeno'] = nil,													-- no cs1|2 equivalent: e-Petition number
		['editörler'] = nil;													-- no cs1|2 equivalent: |editors=
		['editörlerigöster'] = 'display-editors',
		['erişim-tarihi'] = nil,												-- access-date
		['erişimtarihi'] = nil,													-- access-date
		['erişim tarihi'] = nil,												-- access-date
		['erişimyılı'] = nil,													-- access-date
		['eser'] = 'work',
		['eşyazarlar'] = nil,													-- no cs1|2 equivalent: |coauthor(s)=
		['etkinlik'] = 'event',
		['etkinlik-biçimi'] = 'event-format',
		['etkinlik-url'] = 'event-url',
		['farklı-alfabe-başlık'] = 'script-title',
		['farklıalfabebaşlık'] = 'script-title',
		['gazete'] = 'newspaper',
		['girdi-url-erişimi'] = 'entry-url-access',
		['habergrubu'] = 'newsgroup',
		['harita'] = 'map',
		['harita-biçimi'] = 'map-format',
		['harita-url'] = 'map-url',
		['harita-url-erişimi'] = 'map-url-access',
		['ile'] = 'via',
		['istasyon'] = 'station',
		['iş'] = 'work',
		['işbirliği'] = 'collaboration',
		['kanal'] = 'station',
		['katkı'] = 'contribution',
		['katkı-url'] = 'contribution-url',
		['katkı-url-erişimi'] = 'contribution-url-access',
		['katkıdabulunanlar'] = 'authors',										-- |contributors=
		['kaynak'] = 'ref',
		['kayıtlı'] = nil,														-- not in aliases list (registration)
		['kitapbaşlığı'] = 'book-title',
		['kitap-başlığı'] = 'book-title',
		['kişiler'] = 'authors',
		['kısım'] = 'entry',
		['kısım-biçimi'] = 'entry-format',
		['kısım-url'] = 'entry-url',
		['kısım-url-erişimi'] = 'entry-url-access',
		['konferans'] = 'conference',
		['konferans-biçimi'] = 'conference-format',
		['konferans-url'] = 'conference-url',
		['konum'] = 'location',
		['kurum'] = 'network',
		['lisans'] = 'degree',
		['makale'] = 'article',
		['madde'] = 'entry',
		['madde-url-erişimi'] = 'entry-url-access',
		['medya'] = 'medium',
		['mesaj-id'] = 'message-id',
		['muhatap'] = nil,														-- no cs1|2 equivalent: used in {{E-posta kaynağı}} (e-mail source)
		['muhataplar'] = nil,													-- no cs1|2 equivalent: used in {{E-posta kaynağı}} (e-mail source)
		['muhatapbağı'] = nil,													-- no cs1|2 equivalent: used in {{E-posta kaynağı}} (e-mail source)
		['numara'] = 'number',
		['ortakyazarlar'] = nil,												-- no cs1|2 equivalent: |coauthor(s)=
		['ortam'] = 'medium',
		['ölüurl'] = nil,														-- special case: |url-status=
		['ölü-url'] = nil,														-- special case: |url-status=
		['özgünyıl'] = nil,														-- orig-date; is it proper to translate this? what about non-date text?
		['posta-listesi'] = 'mailing-list',			-- posta listesi kaynağı için
		['postalistesi'] = nil,													-- not in aliases list; |mailing-list=
		['program'] = nil,														-- no cs1|2 equivalent
		['s'] = nil,															-- no cs1|2 equivalent: source?
		['sayfa'] = 'page',
		['ss'] = 'pages',
		['ssyok'] = 'no-pp',
		['sayfalar'] = 'pages',
		['sayı'] = 'issue',
		['sene'] = nil,															-- year
		['seri'] = 'series',
		['seribağı'] = 'series-link',
		['seribağlantısı'] = 'series-link',
		['seri-bağlantısı'] = 'series-link',
		['seri-numarası'] = 'series-number',
		['sezon'] = 'season',
		['sınıf'] = 'class',				-- arxiv ve arxiv kaynakları için
		['sonyazarve'] = nil,													-- special case: |name-list-style=amp
		['sözlük'] = 'dictionary',
		['süreliyayın'] = 'periodical',
		['sürüm'] = 'series',
		['şebeke'] = 'network',
		['şehir'] = nil,														-- no cs1|2 equivalent: |city=
		['tanıtıcı'] = 'id',
		['tarih'] = nil,														-- date
		['tip'] = 'type',
		['tür'] = 'type',
		['url-erişimi'] = nil,													-- special case |url-access=
		['url-erişimi'] = nil,													-- special case |url-access=
		['url-durumu'] = 'url-status',
		['websitesi'] = 'website',
		['versiyon'] = 'version',
		['yazarlar'] = 'authors',
		['yardımcıyazarlar'] = nil,												-- no cs1|2 equivalent: |coauthor(s)=
		['yardımcı yazarlar'] = nil,											-- no cs1|2 equivalent: |coauthor(s)=
		['yayın-tarihi'] = nil,													-- publication-date
		['yayıntarihi'] = nil,													-- publication-date
		['yayın-yeri'] = 'publication-place',
		['yayınyeri'] = 'publication-place',
		['yayıncı'] = 'publisher',
		['yayımcı'] = 'publisher',
		['yazars'] = 'authors',
		['yazarları-göster'] = 'display-authors',								-- accepts cs1|2 values: digits or 'etal' string so these are simple translations
		['yazarlarıgöster'] = nil,												-- not in aliases list; |display-authors=
		['yazar-göster'] = 'display-authors',
		['yazar-ad'] = nil,														-- not in aliases list; |author-first=?
		['yazar-soyadı'] = nil,													-- not in aliases list; |author-last=?
		['yer'] = 'location',
		['yıl'] = nil,															-- year
		['zaman'] = 'time',

	-- enumerated parameters; non-enumerated forms of these parameters created by build_params_main_t()
		['ad#'] = 'first#',
		['çevirmen#-ad'] = 'translator#-first',
		['çevirmen#-soyadı'] = 'translator#-last',
		['çevirmen#'] = 'translator#',
		['çevirmen#-bağ'] = 'translator#-link',
		['çevirmen#-bağı'] = 'translator#-link',
		['çevirmen#-maskesi'] = 'translator#-mask',
		['çevirmen#-maske'] = 'translator#-mask',
		['editör#'] = 'editor#',
		['editör#-ilk'] = 'editor#-first',
		['editör#-ad'] ='editor#-first',
		['editör#-bağ'] = 'editor#-link',
		['editör#-bağı'] = 'editor#-link',
		['editör#-son'] = 'editor#-last',
		['editör#-soyadı'] = 'editor#-last',
		['editör#-maskesi'] = 'editor#-mask',
		['editör#-maske'] = 'editor#-mask',
		['görüşmeci#'] = 'interviewer#',
		['görüşmeci#-ad'] = 'interviewer#-first',
		['görüşmeci#-bağ'] = 'interviewer#-link',
		['görüşmeci#-bağlantı'] = 'interviewer#-link',
		['görüşmeci#-maske'] = 'interviewer#-mask',
		['görüşmeci#-maskesi'] = 'interviewer#-mask',
		['görüşmeci#-soyadı'] = 'interviewer#-last',
		['ilk#'] = 'first#',
		['katkı-ad#'] = 'contributor-first#',
		['katkı-soyadı#'] = 'contributor-last#',
		['katkı-bağı#'] = 'contributor-link#',
		['katkı-maskesi#'] = 'contributor-mask#',
		['katkı#-ad'] = 'contributor#-first',
		['katkı#-soyadı'] = 'contributor#-last',
		['katkı#-bağ'] = 'contributor#-link',
		['katkı#-maske'] = 'contributor#-mask',
		['konu#'] = 'subject#',
		['konubağı#'] = 'subject-link#',
		['muhatapadı#'] = nil,													-- no cs1|2 equivalent: |addressee#=? used in {{E-posta kaynağı}} (e-mail source)
		['muhatapbağı#'] = nil,													-- no cs1|2 equivalent: |businesspartner#=?
		['muhatapsoyadı#'] = nil,												-- no cs1|2 equivalent: |addresseename#=? used in {{E-posta kaynağı}} (e-mail source)
		['özne#'] = 'subject#',
		['öznebağı#'] = 'subject-link#',
		['son#'] = 'last#',
		['soyadı#'] = 'last#',
		['süje#'] = 'subject#',
		['süjebağı#'] = 'subject-link#',
		['yazar#'] = 'author#',
		['yazarbağı#'] = 'author-link#',
		['yazarlink#'] = 'author-link#',
		['yazarmaskesi#'] = 'author-mask#',
		['yazar#bağ'] = 'author#-link',
		['yazar#link'] = 'author#-link',
		['yazar#-bağ'] = 'author#-link',
		['yazar#-link'] = 'author#-link',
		},
	}


--[[--------------------------< P A R A M S _ D A T E S _ T >--------------------------------------------------

<date_params_t> is a k/v_t table where k is the the Wikimedia subdomain (language code; 'en' in 'en.wikipedia.org')
and v_t is a k/v_t table where k identifies the 'date' or 'date-part' and v_t is a sequence table of associated
non-English parameter alias names

non-English parameter names are normalized to lowercase.

]]

local params_dates_t = {
	ca = {																		-- Catalan
		date_t = {'data'},
		year_t = {'any'},
		month_t = {'mes'}
		},

	da = {																		-- Danish
		date_t = {'dato', 'datum', 'dag', 'date'},
		year_t = {'år', 'year'},
		},

	de = {																		-- German
		date_t = {'datum', 'date'},
		year_t = {'jahr', 'datum-jahr', 'year'},
		month_t = {'monat'},
		day_t = {'tag'},
		},

	es = {																		-- Spanish
		date_t = {'fecha', 'date'},
		year_t = {'año', 'year'},
		},

	fi = {																		-- Finnish
		date_t = {'ajankohta', 'date'},
		year_t = {'vuosi', 'year'},
		},

	fr = {																		-- French
		date_t = {'date'},
		year_t = {'année', 'annee', 'year'},
		month_t = {'mois'},
		day_t = {'jour'},
		},

	it = {																		-- Italian
		date_t = {'data', 'datatrasmissione', 'date'},
		year_t = {'anno', 'year'},
		month_t = {'mese'},
		day_t = {'giorno'},
		},

	nl = {																		-- Dutch
		date_t = {'datum', 'date'},
		year_t = {'jaar', 'year'},
		month_t = {'maand', 'month'},
		day_t = {'dag'},
		},

	no = {																		-- Norwegian
		date_t = {'dato', 'date'},
		year_t = {'år', 'utgivelsesår', 'year'},
		month_t = {'måned'},
		day_t = {'dag'},
		},

	pl = {																		-- Polish
		date_t = {'data', 'date'},
		year_t = {'rok', 'year'},
		month_t = {'miesiąc'},
		},

	pt = {																		-- Polish
		date_t = {'data', 'dataemissao', 'fecha', 'date'},
		year_t = {'ano', 'año', 'year'},
		month_t = {'acessomesdia', 'mes', 'month', 'mês'},
		},

	ru = {																		-- Russian
		year_t =  {'год', 'year'},												-- also date
		month_t = {'месяц'},
		day_t = {'число'},
		},

	sv = {																		-- Swedish
		date_t = {'date', 'publdatum', 'datum'},
		year_t = {'år', 'year'},
		month_t = {'månad'},
		day_t = {'dag'},
		},

	tr = {																		-- Turkish
		date_t = {'tarih', 'date'},
		year_t = {'yıl', 'year'},
		month_t = {'ay'},
		day_t = {'gün'},
		},
	}


--[[--------------------------< P A R A M S _ M I S C _ D A T E S _ T >----------------------------------------

For date-holding parameters that are not |date=, |year=, |month=, or |day= equivalents.

table of k/v_t pairs where k/v_t in the outer table is:
	k – the Wikimedia subdomain (language code; 'en' in 'en.wikipedia.org')
	v_t – a table of k/v pairs where
		k – the non-English parameter name
		v – the directly translatable cs1|2 equivalent date-holding parameters that are not |date=, |year=, |month=,
			or |day= equivalents.

All key values normalized to lowercase.

]]

local params_misc_dates_t = {
	ca = {																		-- Catalan
		['arxiudata'] = 'archive-date',
		['consulta'] = 'access-date',
		['dataaccés'] = 'access-date',
		},

	da = {																		-- Danish
		['accessdate'] = 'access-date',
		['arkivdato'] = 'archive-date',
		['besøgsdato'] = 'access-date',
		['hentningsdato'] = 'access-date',
		['hentet'] = 'access-date',
		['hämtdatum'] = 'access-date',
		['besøksdato'] = 'access-date',
		['accesdate'] = 'access-date',
		['accessed'] = 'access-date',
		['acces date'] = 'access-date',
		['acessdate'] = 'access-date',
		['acces-date'] = 'access-date',
		['accessdato'] = 'access-date',
		['hentedag'] = 'access-date',
		['originalår'] = 'orig-date',											-- is it proper to translate this? what about non-date text?
		},

	de = {																		-- German
		['abruf'] = 'access-date',
		['zugriff'] = 'access-date',
		['zugriff-jahr'] = 'access-date',
		['archiv-datum'] = 'archive-date',
		},

	en = {																		-- en.wiki cannonical and alternate forms to catch partial translations
		['accessdate'] = 'access-date',											-- en.wiki alternate form
		['access-date'] = 'access-date',
		['archive-date'] = 'archive-date',
		['doi-broken-date'] = 'doi-broken-date',
		['lay-date'] = 'lay-date',
		['publication-date'] = 'publication-date',
		},

	es = {																		-- Spanish
		['año-original'] = 'orig-date',											-- orig-date; is it proper to translate this? what about non-date text?
--		['añoacceso'] = 'access-date',											-- no cs1|2 equivalent: |access-year=
		['doibroken'] = 'doi-broken-date',										-- no longer supported by cs1|2
		['doi_brokendate'] = 'doi-broken-date',									-- no longer supported by cs1|2
		['doi_inactivedate'] = 'doi-broken-date',								-- no longer supported by cs1|2
		['fechaacceso'] = 'access-date',
		['fechaarchivo'] = 'archive-date',
		['fecha-acceso'] = 'access-date',
		['fecha-doi-roto'] = 'doi-broken-date',
		['fechaprofano'] = 'lay-date',
		['fecha-publicación'] = 'publication-date',
		['fecharesumen'] = 'lay-date',
		['fecha-resumen'] = 'lay-date',
		['laydate'] = 'lay-date',												-- no longer supported by cs1|2
		['publicationdate'] = 'publication-date',								-- no longer supported by cs1|2
		},

	fi = {																		-- Finnish
		['arkistoitu'] = 'archive-date',
		['luettu'] = 'access-date',
		['viitattu'] = 'access-date',
		},

	fr = {																		-- French
		['archivedate'] = 'archive-date',
		['consulte le'] = 'access-date',
		['consulté le'] = 'access-date',
		},

	it = {																		-- Italian
		['accesso'] = 'access-date',
--		['annoaccesso'] = 'access-date',										-- no cs1|2 equivalent: |access-year=
--		['annodiaccesso'] = 'access-date',										-- no cs1|2 equivalent: |access-year=
		['annooriginale'] = 'orig-date',										-- |orig-year= only; |meseoriginale= (|orig-month=) and |giornooriginale= (|orig-day=) not supported
		['dataaccesso'] = 'access-date',
		['dataarchivio'] = 'archive-date',
		['datadiaccesso'] = 'access-date',
		['dataoriginale'] = 'orig-date',										-- orig-date; is it proper to translate this? what about non-date text?
		['datapubblicazione'] = 'publication-date',
		['datatrasmissione'] = 'date',
		['doibroken'] = 'doi-broken-date',										-- no longer supported by cs1|2
		['doi_brokendate'] = 'doi-broken-date',									-- no longer supported by cs1|2
		['doi_inactivedate'] = 'doi-broken-date',								-- no longer supported by cs1|2
--		['giornoaccesso'] = 'access-date',										-- no cs1|2 equivalent: |access-day=
--		['giornodiaccesso'] = 'access-date',									-- no cs1|2 equivalent: |access-day=
--		['giornooriginale'] = 'orig-date',										-- no cs1|2 equivalent: |orig-day=
		['laydate'] = 'lay-date',												-- no longer supported by cs1|2
--		['meseaccesso'] = 'access-date',										-- no cs1|2 equivalent: |access-month=
--		['mesediaccesso'] = 'access-date',										-- no cs1|2 equivalent: |access-month=
--		['meseoriginale'] = 'orig-date',										-- no cs1|2 equivalent: |orig-month=
		},

	nl = {																		-- Dutch
		['accessdate'] = 'access-date',
		['archivedate'] = 'archive-date',
		['archiefdatum'] = 'archive-date',
		['bezochtdatum'] = 'access-date',
		['datumbezocht'] = 'access-date',
		['datumgeraadpleegd'] = 'access-date',
		['publicatiedatum'] = 'publication-date',
		['raadpleegdatum'] = 'access-date',
		},

	no = {																		-- Norwegian
		['arkivdato'] = 'archive-date',
		['arkiv_dato'] = 'archive-date',
		['arkiv-dato'] = 'archive-date',
		['doibrutt'] = 'doi-broken-date',
		['doi_bruttdato'] = 'doi-broken-date',
		['doi_inaktivdato'] = 'doi-broken-date',
		['embargo'] = 'pmc-embargo-date',
		['opprinnelsesår'] = 'orig-date',										-- orig-date; is it proper to translate this? what about non-date text?
		['sammendragdato'] = 'lay-date',
		['utgivelsesdato'] = 'publication-date',
		['utgivelses-dato'] = 'publication-date',
		},

	pl = {																		-- Polish
		['data dostępu'] = 'access-date',
		['zarchiwizowano'] = 'archive-date',
		},

	pt = {																		-- Polish
		['accessodata'] = 'access-date',
		['acessadoem'] = 'access-date',
		['acessdate'] = 'access-date',
		['acesso'] = 'access-date',
		['acesso-data'] = 'access-date',
		['acessodata'] = 'access-date',
		['anooriginal'] = 'orig-date',
		['arquivo-data'] = 'archive-date',
		['arquivodata'] = 'archive-date',
		['consulta'] = 'access-date',
		['data-publicacao'] = 'publication-date',
		['data-publicação'] = 'publication-date',
		['dataacesso'] = 'access-date',
		['fecha-publicación'] = 'publication-date',
		['fechaacceso'] = 'access-date',
		['publicationdate'] = 'publication-date',
		['transmissão'] = 'air-date',
		},

	sv = {																		-- Swedish
		['accessdate'] = 'access-date',
		['arkivdatum'] = 'archive-date',
		['date'] = 'access-date',												-- different from en.wiki
		['hämtdatum'] = 'access-date',
		['origår'] = 'orig-date',												-- actually |orig-year=
		['origdatum'] = 'orig-date',
--		['origmånad'] = nil,													-- no cs1|2 equivalent; |orig-month=
		['utgivningsdatum'] = 'publication-date',
		},

	tr = {																		-- Turkish
		['archivedate'] = 'archive-date',										-- no longer supported by cs1|2
		['arşivtarihi'] = 'archive-date',
		['arşiv-tarihi'] = 'archive-date',
		['diğertarih'] = 'lay-date',
		['doibozuk'] = 'doi-broken-date',
		['doi-broken'] = 'doi-broken-date',										-- no longer supported by cs1|2
		['doi-hatalı-tarih'] = 'doi-broken-date',
		['doi-inactive-date'] = 'doi-broken-date',								-- no longer supported by cs1|2
		['doi-kırık-tarih'] = 'doi-broken-date',
		['doi_bozuktarih'] = 'doi-broken-date',
		['doi_brokendate'] = 'doi-broken-date',									-- no longer supported by cs1|2
		['doi_inactivedate'] = 'doi-broken-date',								-- no longer supported by cs1|2
		['erişimtarihi'] = 'access-date',
		['erişim-tarihi'] = 'access-date',
		['erişim tarihi'] = 'access-date',
		['laydate'] = 'lay-date',												-- no longer supported by cs1|2
		['origyear'] = 'orig-date',												-- orig-date; is it proper to translate this? what about non-date text?
		['özgünyıl'] = 'orig-date',												-- orig-date; is it proper to translate this? what about non-date text?
		['yayıntarihi'] = 'publication-date',
		['yayın-tarihi'] = 'publication-date',
		}
	}


--[[--------------------------< P A R A M S _ I D E N T I F I E R S _ T >--------------------------------------

miscellaneous identifiers that, at en.wiki, are grouped together in |id=

table of k/v_t pairs where k/v_t in the outer table is:
	k – the Wikimedia subdomain (language code; 'en' in 'en.wikipedia.org')
	v_t – a sequence table of sequence tables where:
		[1] is the parameter name normalized to lower case
		[2] is the associated wikitext label to be used in the rendering
		[3] is the url-prefix to be attached to the identifier value from the template parameter
		[4] is the url-postfix to be attached to the identifier value

parameter names are normalized to lowercase.

]]

local params_identifiers_t = {															-- identifier parameters (|ID=, |URN=, etc) and their associated labels for inclusion in |id=
	de = {																		-- German
		{'id'},																	-- |id= does not get a label so nil
		{'urn', '[[Uniform Resource Name|URN]]'},
		{'dnb', '[[DNB-IDN (identifier)|DNB-IDN]]', 'http://d-nb.info/'},
		{'zdb', '[[ZDB-ID (identifier)|ZDB-ID]]', 'http://ld.zdb-services.de/resource/'},
		},

	fr = {																		-- french
		{'bnf', '[[BNF (identifier)|BNF]]', 'http://catalogue.bnf.fr/ark:/12148/cb', '.public'},	-- has a postfix
		{'dnb', '[[DNB-IDN (identifier)|DNB-IDN]]', 'http://d-nb.info/'},
		{'ean', '[[EAN (identifier)|EAN]]'},
		{'hal', '[[HAL (open archive)|HAL]]'},
		{'libris', [[LIBRIS]]},
		{'sudoc', [[SUDOC (identifier)|SUDOC]]}
		},

	nl = {																		-- Dutch
		{'nur', 'NUR'},															-- [[:nl:Nederlandstalige Uniforme Rubrieksindeling]]
		},

	sv = {																		-- Swedish
		{'libris', '[[LIBRIS]]', 'http://libris.kb.se/bib/'},
		},
	}


--[[--------------------------< P A R A M S _ L A N G U A G E _ T >--------------------------------------------

table of k/v_t pairs where k/v_t in the outer table is:
	k – the Wikimedia subdomain (language code; 'en' in 'en.wikipedia.org')
	v_t – a sequence table of non-English equivalents to the en.wiki |language= parameter

]]

local params_language_t = {
	ca = {'idioma', 'llengua'},													-- Catalan
	da = {'sprog', 'på', 'språk', 'langue', 'lang'},							-- Danish
	de = {'originalsprache', 'sprache'},										-- German
	es = {'idioma', 'language'},												-- Spanish
	fi = {'kieli', 'language'},													-- Finnish
	fr = {'langue', 'language'},												-- French
	it = {'lingua'},															-- Italian
	nl = {'språk', 'language'},													-- Dutch
	no = {'på', 'språk'},														-- Norwegian
	ru = {'язык'},																-- Russian
	sv = {'språk', 'language'},													-- Swedish
	tr = {'dil', 'language'},													-- Turkish
	}


--[[--------------------------< B U I L D _ P A R A M S _ M A I N _ T >----------------------------------------

assemble the main list of parameters; skip all nil-valued parameter and create non-enumerated parameter names
from the enumerated parameters (those that have '#' somewhere in the parameter name)

Does simple error detection and emits a graringly crude error message when:
	only one side of ['key'] = value pair has a '#'; when enumerated, both sides require the '#'
	the only type allowed for value in a ['key'] = value pair is 'string'; catches things like ['side'] = true (copied from a whitelist)

]]

local function build_params_main_t ()
	local out_t = {};															-- table goes here
	for lang, v_t in pairs (params_main_t) do									-- for each language table in params_main_t{}
		out_t[lang] = {};														-- create a table in out_t for <lang>
		for k, v in pairs (v_t) do												-- for each parameter in the language table
			if 'string' ~= type (v) then
				error (lang .. ' ' .. k .. ' value not a string');				-- glaring error message because non-string values not allowed
			end
			if v then															-- if the parameter has a non-nil translation (not a special, not a parameter without cs1|2 equivalent)
				if (k:find ('#', 1, true) and not v:find ('#', 1, true)) or
					(not k:find ('#', 1, true) and v:find ('#', 1, true))then
						error (lang .. '[' .. k .. ']: '.. v .. ' missing \'#\'');	-- glaring error message because '#' required on both sides
				else
					if k:find ('#', 1, true) then								-- does the parmeter name have the enumerator character '#'?
						out_t[lang][k:gsub('#', '')] = v:gsub('#', '');			-- add a non-enumerated version of the parameter to the output
					end
					out_t[lang][k] = v;											-- add the parameter to the output; may be an enumerated param or not
				end
			end
		end
	end
	return out_t;																-- and done
end


--[[--------------------------< E X P O R T E D   T A B L E S >------------------------------------------------
]]

return {
	params_dates_t = params_dates_t,
	params_identifiers_t = params_identifiers_t,
	params_language_t = params_language_t,
	params_main_t = build_params_main_t(),
	params_misc_dates_t = params_misc_dates_t,
	}