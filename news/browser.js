/**
 * Run the script with node.js:
 *   node browser.js
 * Dependencies: xh2, jsdom
 */

const fs = require("fs");
const XMLHttpRequest = require('xhr2');
const jsdom = require("jsdom")
const { JSDOM } = jsdom
global.DOMParser = new JSDOM().window.DOMParser
const domParser = new DOMParser()

// We use ScraperAPI to get HTML from web pages, see https://www.scraperapi.com/.
// Create an account and provide your API key here
// const apiKey = 'YOUR SCRAPER API KEY';
const {apiKey} = require('./api_key')

function process(data, url, source){
    const results = [];
    let doc = domParser.parseFromString(data, "text/html");
    let urls = doc.getElementsByTagName('a');
    for (urlIndex in urls) {
        const url = urls[urlIndex];
        if(url.href && url.href.indexOf('://')!==-1 &&
            !url.href.startsWith(source) &&
            !url.href.startsWith('file')
        ) {
            results.push([url.href]);
        }
    }
    const csvContent = results.map((line)=>{
        return line.map((cell)=>{
            if(typeof(cell)==='boolean') return cell ? 'TRUE': 'FALSE'
            if(!cell) return ''
            let value = cell.replace(/[\f\n\v]*\n\s*/g, "\n").replace(/[\t\f ]+/g, ' ');
            value = value.replace(/\t/g, ' ').trim();
            return `"${value}"`
        }).join('\t')
    }).join("\n");
    console.log('***' + url);
    console.log(csvContent);
}

function httpGetAsync(source, page, callback){
    const baseUrl = `http://api.scraperapi.com?api_key=${apiKey}&url=`;
    const url = baseUrl+page;
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            callback(xmlHttp.responseText, page, source);
        }
    }
    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function get_data(source, inputFile){
    fs.readFile(inputFile, (err, data) => {
        if (err) throw err;
        const pages = data.toString().split(/\r?\n/);
        pages.forEach(page => {
            httpGetAsync(source, page, process);
        });
    });
}

// TODO Uncomment to collect data from a chosen source

// get_data('https://www.newspaperindex.com', 'newspaperindex/countries.txt')

// get_data('https://www.newsmedialists.com', 'newsmedialists/countries-newspapers.txt')
// get_data('https://www.newsmedialists.com', 'newsmedialists/countries-tv.txt')
get_data('https://www.newsmedialists.com', 'newsmedialists/countries-magazines.txt')



