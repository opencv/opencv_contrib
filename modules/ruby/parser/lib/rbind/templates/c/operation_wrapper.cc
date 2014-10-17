// operation wrapper for <%= attribute? ? "#{owner.full_name}.#{attribute.name}" : signature %>
<%= csignature %>
{
    try
    {
        <%= wrap_parameters %><%= wrap_call %>
    }
    catch(std::exception &error){strncpy(&last_error_message[0],error.what(),255);}
    catch(...){strncpy(&last_error_message[0],"Unknown Exception",255);}
    <%- if !return_type || return_type.ptr? || !return_type.basic_type? -%>
    return NULL;
    <%- elsif return_type.name != "void"  -%>
    return (<%= return_type.cname %>) <%= return_type.invalid_value %>;
    <%- end -%>
}
