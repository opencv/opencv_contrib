// wrapper for <%= full_name %>
typedef struct <%= cname %>
{
    unsigned char version;  // version of the object
    size_t size;            // size of the object in bytes
    void *type_id;          // type id of the object
    void *obj_ptr;          // ptr to <%= full_name %>
    bool bowner;            // true if struct is the owner of the object
}<%= cname %>;

RBIND_EXPORTS void <%= cdelete_method %>(<%= cname %> *ptr);

