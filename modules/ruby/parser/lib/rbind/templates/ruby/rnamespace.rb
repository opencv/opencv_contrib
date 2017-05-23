<%= add_doc -%>
# @note namespace wrapper for <%= full_name %>
module <%= name %>
    # @!group Constants
<%= add_consts%>
    # @!endgroup

<%= add_methods %>
<%= add_types %>
end

