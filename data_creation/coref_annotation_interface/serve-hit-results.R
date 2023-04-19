library(shiny)
library(tidyverse)
library(jsonlite)

ui <- fluidPage(
    tags$head(
        tags$style(HTML("
            .text-btn { padding: 1px; }
            .text-container { margin-bottom: 1em; }
            
            .btn-danger { background-color: #ac2925; }
            .btn-info { background-color: #269abc; }
            .btn-warning { background-color: #d58512; }
            .btn-success { background-color: #398439; }
            
            .btn-danger.btn-info {
              background-image: linear-gradient(45deg, #ac2925 25%, #269abc 25%, #269abc 50%, #ac2925 50%, #ac2925 75%, #269abc 75%, #269abc 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-danger.btn-warning {
              background-image: linear-gradient(45deg, #ac2925 25%, #d58512 25%, #d58512 50%, #ac2925 50%, #ac2925 75%, #d58512 75%, #d58512 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-info.btn-warning {
              background-image: linear-gradient(45deg, #269abc 25%, #d58512 25%, #d58512 50%, #269abc 50%, #269abc 75%, #d58512 75%, #d58512 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-danger.btn-info.btn-warning {
              background-image: linear-gradient(45deg, #ac2925 16.67%, #269abc 16.67%, #269abc 33.33%, #d58512 33.33%, #d58512 50%, #ac2925 50%, #ac2925 66.67%, #269abc 66.67%, #269abc 83.33%, #d58512 83.33%, #d58512 100%);
            background-size: 29.70px 29.70px;
            }
                        
            .btn-danger.btn-success {
              background-image: linear-gradient(45deg, #ac2925 25%, #398439 25%, #398439 50%, #ac2925 50%, #ac2925 75%, #398439 75%, #398439 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-danger.btn-warning {
              background-image: linear-gradient(45deg, #ac2925 25%, #d58512 25%, #d58512 50%, #ac2925 50%, #ac2925 75%, #d58512 75%, #d58512 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-success.btn-warning {
              background-image: linear-gradient(45deg, #398439 25%, #d58512 25%, #d58512 50%, #398439 50%, #398439 75%, #d58512 75%, #d58512 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-danger.btn-success.btn-warning {
              background-image: linear-gradient(45deg, #ac2925 16.67%, #398439 16.67%, #398439 33.33%, #d58512 33.33%, #d58512 50%, #ac2925 50%, #ac2925 66.67%, #398439 66.67%, #398439 83.33%, #d58512 83.33%, #d58512 100%);
            background-size: 29.70px 29.70px;
            }
                        
            .btn-danger.btn-success {
              background-image: linear-gradient(45deg, #ac2925 25%, #398439 25%, #398439 50%, #ac2925 50%, #ac2925 75%, #398439 75%, #398439 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-danger.btn-info {
              background-image: linear-gradient(45deg, #ac2925 25%, #269abc 25%, #269abc 50%, #ac2925 50%, #ac2925 75%, #269abc 75%, #269abc 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-success.btn-info {
              background-image: linear-gradient(45deg, #398439 25%, #269abc 25%, #269abc 50%, #398439 50%, #398439 75%, #269abc 75%, #269abc 100%);
            background-size: 28.28px 28.28px;
            }
            
            .btn-danger.btn-success.btn-info {
              background-image: linear-gradient(45deg, #ac2925 16.67%, #398439 16.67%, #398439 33.33%, #269abc 33.33%, #269abc 50%, #ac2925 50%, #ac2925 66.67%, #398439 66.67%, #398439 83.33%, #269abc 83.33%, #269abc 100%);
            background-size: 29.70px 29.70px;
            }"))
    ),
    
    titlePanel("UDS Training Annotations"),
    # titlePanel("UDS Training Annotations (Outliers)"), #OUTLIERS

    fluidRow(
        column(12,
               div(list(actionButton("new_hit", "Sample New HIT"), span(verbatimTextOutput("title")))),
               uiOutput("query_tabs")
        )
    )
)

load_data <- function() {
    datetime_format <- "%.%.%. %b %e %H:%M:%S %.%.%. %Y"
    mturk_cols <- cols(
        CreationTime=col_datetime(format=datetime_format),
        AcceptTime=col_datetime(format=datetime_format),
        SubmitTime=col_datetime(format=datetime_format)
    )
    annotators <- c(
        'GusRodriguez',
        'JamiesonAlexander',
        'JoselynCarretero',
        'JTWilson',
        'NataliaMauroni'
    )
    split_inputs <- c('train', 'dev', 'test') %>%
        map_dfr(~ tibble(Input.json_data=read_lines(str_c('serve-hit-results-data/uds-', .x, '-hits.jsonl')), split=.x))
    annotators %>%
        map_dfr(~ read_csv(str_c('serve-hit-results-data/', .x, '-results.csv'), col_types=mturk_cols)) %>%
        mutate(Answer.answer_spans=Answer.answer_spans %>% map(~ fromJSON(.x, flatten=T))) %>%
        left_join(split_inputs, by='Input.json_data') %>%
        unnest(Answer.answer_spans) %>%
        mutate(query_id=group_indices(., Input.json_data, querySpan.sentenceIndex, querySpan.startToken, querySpan.endToken),
               answer_id=group_indices(., notPresent, sentenceIndex, startToken, endToken),
               split=factor(split, levels=c('train', 'dev', 'test'))) %>%
        filter(split == 'train') # %>% #OUTLIERS
        # filter(WorkTimeInSeconds > 300) #OUTLIERS
}

ensure_list <- function(x) {
    if (is.list(x)) { x } else { list(x) }
}

color_map <- tibble(
    annotator_num=c(1, 2, 3),
    button_class=c("btn-danger", "btn-warning", "btn-info")
)

liftedNavlistPanel <- lift(navlistPanel)

server <- function(input, output, session) {
    full_data <- withProgress({
        setProgress(message = "Loading data...")
        load_data()
    })
    
    sample_data <- reactive({
        input$new_hit
        full_data %>%
            filter(Input.json_data == (Input.json_data %>% unique %>% sample(1)))
    })
    
    input_data <- reactive({
        sample_data()$Input.json_data[1] %>% fromJSON
    })
    
    sentence_ids <- reactive({
        input_data()$sentenceIds
    })
    
    sentences <- reactive({
        input_data()$sentences %>% ensure_list
    })
    
    output$title <- renderText({
        sentence_ids() %>% last
    })
    
    output$query_tabs <- renderUI({
        withProgress({
            setProgress(message = "Formatting data...")
            sents <- sentences()
            sample_data() %>%
                mutate(
                    annotator_num=as.numeric(factor(WorkerId)),
                    query_text=pmap_chr(
                        list(querySpan.sentenceIndex, querySpan.startToken, querySpan.endToken),
                        ~ paste(sents[[..1 + 1]][(..2 + 1):..3], collapse=' '))) %>%
                left_join(color_map) %>%
                arrange(querySpan.sentenceIndex, querySpan.startToken, querySpan.endToken) %>%
                nest(-query_id, -query_text) %>%
                mutate(query_num=row_number()) %>%
                pmap(function(query_id, query_num, query_text, data) {
                    tabPanel(
                        query_text,
                        1:length(sents) %>% map(function(sentence_num) {
                            sentence <- sents[[sentence_num]]
                            span(1:length(sentence) %>% map(function(word_num) {
                                word <- sentence[word_num]
                                word_data <- data %>%
                                    mutate(
                                        word_in_query=(
                                            (sentence_num-1) == querySpan.sentenceIndex &
                                                (word_num-1) >= querySpan.startToken &
                                                (word_num-1) < querySpan.endToken),
                                        word_in_answer=(
                                            (sentence_num-1) == sentenceIndex &
                                                (word_num-1) >= startToken &
                                                (word_num-1) < endToken))
                                query_class <- if (word_data$word_in_query %>% first) {
                                    'btn-primary'
                                } else {
                                    ''
                                }
                                answer_class <- paste(
                                    (word_data %>% filter(word_in_answer))$button_class,
                                    collapse=' ')
                                span(class=paste('btn text-btn', query_class, answer_class), word)
                            }))}) %>%
                            div(class='text-container'),
                        data %>%
                            filter(notPresent) %>%
                            pmap(function(button_class, ...) {
                                span(class=paste('btn text-btn', button_class), "not present")
                            }) %>%
                            div
                    )}) %>%
                c(list('Queries', widths=c(2, 10)), .) %>%
                liftedNavlistPanel
        })
    })
}

shinyApp(ui, server)