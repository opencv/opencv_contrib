<?php
function endsWith( $haystack, $needle ) {
    $length = strlen( $needle );
    if( !$length ) {
        return true;
    }
    return substr( $haystack, -$length ) === $needle;
}

// open the file in a binary mode
$resource = $_GET["res"];
$name = "./${resource}";
$fp = fopen($name, 'rb');

// send the right headers
if(endsWith($resource, ".html")) {
	header("Content-Type: text/html");
} elseif (endsWith($resource, ".js")) {
	header("Content-Type: text/javascript");
} elseif (endsWith($resource, ".wasm")) {
        header("Content-Type: application/wasm");
} else {
        header("Content-Type: text/html");
}
header("Content-Length: " . filesize($name));
header("Cross-Origin-Embedder-Policy: require-corp");
header("Cross-Origin-Opener-Policy: same-origin");
fpassthru($fp);
exit;

?>

