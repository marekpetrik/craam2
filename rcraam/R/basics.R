.normalize <- function(x) {x / sum(x)}

inventory.default <- function(){
    list(
        variable_cost = 2.49,
        fixed_cost = 5.99,
        holding_cost = 0.03,
        backlog_cost = 0.15,
        sale_price = 3.99,
        max_inventory = 100,
        max_backlog = 10,
        max_order = 50,
        demands = .normalize(dnorm(0:20,10,5)),
        seed = 1984)

}

