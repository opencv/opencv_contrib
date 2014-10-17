// converts <%= full_name %>* to <%= cname %>*

const <%= cname %>* toC(const <%= full_name %>* ptr, bool owner)
{
    <%= cname %>* r_ptr = new <%= cname %>;
    r_ptr->version = <%= version %>;
    r_ptr->size = sizeof(*ptr);
    r_ptr->type_id = (void*) &typeid(*ptr);
    r_ptr->bowner = owner;
    r_ptr->obj_ptr = (void*) ptr;
    return r_ptr;
}

<%= cname %>* toC(<%= full_name %>* ptr, bool owner)
{
    return const_cast<<%= cname %>*>(toC(static_cast<const <%= full_name %>*>(ptr),owner));
}

// converts const <%= cname %> to const <%= full_name %>
const <%= full_name %>* fromC(const <%= cname %>* ptr)
{
    if(ptr == NULL)
        throw std::runtime_error("<%= full_name %>: Null Pointer!");
    if(!ptr->obj_ptr)
        return NULL;
    <%- if check_type? -%>
    // check typeid if available
    if(ptr->type_id && typeid(<%= full_name %>) != *static_cast<const std::type_info*>(ptr->type_id))
    {
        std::string str("wrong object type for <%= full_name %>: got ");
        throw std::runtime_error(str + static_cast<const std::type_info*>(ptr->type_id)->name()
                    + " but was expecting " + typeid(<%= full_name %>).name());
    }
    // check version
    if(ptr->version != <%= version %>)
        throw std::runtime_error("wrong object version for <%= full_name %>");
    <%- end %>
    // check size
    if(ptr->size && sizeof(<%= full_name %>) > ptr->size)
        throw std::runtime_error("wrong object size for <%= full_name %>.");
    return static_cast<const <%= full_name %>*>(ptr->obj_ptr);
}

// converts <%= cname %>* to <%= full_name %>*
<%= full_name %>* fromC(<%= cname %>* ptr)
{
    if(ptr == NULL)
        return NULL;
    return const_cast<<%= full_name %>*>(fromC(static_cast<const <%= cname %>*>(ptr)));
}

