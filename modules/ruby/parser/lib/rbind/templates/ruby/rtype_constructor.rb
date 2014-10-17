        # wrapper for type constructor <%= signature %>
        if args.size == <%= parameters.size %>
            return Rbind::<%= cname %>(*args)
        end
