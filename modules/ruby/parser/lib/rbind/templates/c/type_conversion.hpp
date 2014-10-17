// convert functions for <%= full_name %>* to and from <%= cname %>*
const <%= cname %>* toC(const <%= full_name %>* ptr, bool owner = true);
<%= cname %>* toC(<%= full_name %>* ptr, bool owner = true);
const <%= full_name %>* fromC(const <%= cname %>* ptr);
<%= full_name %>* fromC(<%= cname %>* ptr);

