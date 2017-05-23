<%= add_doc -%>
    # @note method wrapper for <%= signature %>
    def <%=name%>(<%= wrap_parameters_signature %>)
<%= add_specialize_ruby -%>
        __validate_pointer__
    <%- if return_type.basic_type? || operator? -%>
        Rbind::<%= cname %>( <%= wrap_parameters_call %>)
    <%- else -%>
        result = Rbind::<%= cname %>( <%= wrap_parameters_call %>)
        if result.respond_to?(:__owner__?) && !result.__owner__?
        # store owner insight the pointer to not get garbage collected
            result.instance_variable_get(:@__obj_ptr__).instance_variable_set(:@__owner__,self) 
        end
        result
    <%- end -%>
    end
<%= add_alias -%>

