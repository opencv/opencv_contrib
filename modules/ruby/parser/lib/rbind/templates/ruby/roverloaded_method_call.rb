        # overloaded method wrapper for <%= signature %>
        @@<%=cname%>_defaults<%= index %> ||= <%= signature_default_values %>
        if(args.size >= <%= min_number_of_parameters %> && args.size <= <%= parameters.size %>)
            targs = args.clone
            targs.size.upto(<%= parameters.size-1%>) do |i|
                targs[i] = @@<%=cname%>_defaults<%=index%>[i]
            end
            begin
    <%= add_specialize_ruby -%>
                <%- if !return_type || return_type.basic_type? || operator? -%>
                    <%- if constructor? || !instance_method? -%>
                return Rbind::<%= cname %>(*targs)
                    <%- else -%>
                return Rbind::<%= cname %>(self,*targs)
                    <%- end -%>
                <%- else -%>
                    <%- if instance_method? -%>
                result = Rbind::<%= cname %>(self,*targs)
                    <%- else -%>
                result = Rbind::<%= cname %>(*targs)
                    <%- end -%>
                # store owner insight the pointer to not get garbage collected
                result.instance_variable_get(:@__obj_ptr__).instance_variable_set(:@__owner__,self) if !result.__owner__?
                return result
                <%- end -%>
            rescue TypeError => e
                @error = e
            end
        end
