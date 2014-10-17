<%= add_doc -%>
    # @note wrapper for overloaded static method <%= name %>
    def self.<%=name%>(*args)
<%= add_methods %>
        raise ArgumentError, "No overloaded signature fits to: #{args.map(&:class)}"
    end
<%= add_alias -%>
