/* A script that can be run in a browser console to pick up country links. */

let results = [];
let urls = document.getElementsByTagName('a');
for (urlIndex in urls) {
    const url = urls[urlIndex];
    if(url.href && url.href.indexOf('://')!==-1) {
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
console.log(csvContent);