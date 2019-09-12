library(shiny)
source("gambling.R")

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Gambler's Luck"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            sliderInput("capital", "Max capital:", min = 1, max = 100, value = 99),
            sliderInput("pwin", "Probability win:", min = 0, max = 1, value = 0.4),
            sliderInput("discount", "Discount factor:", min = 0, max = 1, value = 1)         
        ),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("polPlot")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

    output$polPlot <- renderPlot({
        
        sol <- solve.gambling(input$capital, input$pwin, input$discount)
        
        policy.plot <- ggplot(sol$opt.policy, aes(x=idstate,y=idaction,fill=1)) + 
            geom_abline(intercept = 0, slope = 1, color="red") +
            theme(legend.position = "none")+
            geom_tile() + labs(x="State",y="Action")
        policy.plot
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
